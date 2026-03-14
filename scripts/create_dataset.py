import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

in_dir  = "../data"
out_dir = "../Dataset"

os.makedirs(out_dir, exist_ok=True)
print(f"Input  directory : {in_dir}")
print(f"Output directory : {out_dir}")

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = fs / 2
    low     = lowcut  / nyquist
    high    = highcut / nyquist
    try:
        b, a     = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered
    except Exception:
        return signal
    
def load_signal(filepath):
    rows = []
    data_started = False
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == 'Data:':
                data_started = True
                continue
            if not data_started or line == '':
                continue
            parts = line.split(';')
            if len(parts) < 2:
                continue
            timestamp_str = parts[0].strip().replace(',', '.')
            value_str     = parts[1].strip()
            try:
                ts  = pd.to_datetime(timestamp_str, format='%d.%m.%Y %H:%M:%S.%f')
                val = float(value_str)
                rows.append({'time': ts, 'value': val})
            except Exception:
                continue
    return pd.DataFrame(rows)


def load_patient_data(participant_path):
    flow_path   = os.path.join(participant_path, 'nasal_airflow.txt')
    thorac_path = os.path.join(participant_path, 'thoracic_movement.txt')
    spo2_path   = os.path.join(participant_path, 'spo2.txt')

    df_flow   = load_signal(flow_path)
    df_thorac = load_signal(thorac_path)
    df_spo2   = load_signal(spo2_path)

    flow_filt   = bandpass_filter(df_flow['value'].values,   0.17, 0.4, fs=32)
    thorac_filt = bandpass_filter(df_thorac['value'].values, 0.17, 0.4, fs=32)
    spo2_filt   = bandpass_filter(df_spo2['value'].values,   0.17, 0.4, fs=4)

    return df_flow['time'].values, flow_filt, thorac_filt, spo2_filt, df_flow, df_spo2

def load_events(filepath):
    rows = []
    header_done = False
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                header_done = True
                continue
            if not header_done:
                continue
            parts = line.split(';')
            if len(parts) < 3:
                continue
            time_range = parts[0].strip()
            event_type = parts[2].strip()
            dash_idx   = time_range.rfind('-')
            if dash_idx == -1:
                continue
            start_str = time_range[:dash_idx].strip().replace(',', '.')
            end_str   = time_range[dash_idx+1:].strip().replace(',', '.')
            try:
                start_dt  = pd.to_datetime(start_str, format='%d.%m.%Y %H:%M:%S.%f')
                date_part = start_str.split(' ')[0]
                end_dt    = pd.to_datetime(date_part + ' ' + end_str,
                                           format='%d.%m.%Y %H:%M:%S.%f')
                if end_dt < start_dt:
                    end_dt += pd.Timedelta(days=1)
                rows.append({'start': start_dt, 'end': end_dt, 'event': event_type})
            except Exception:
                continue
    return pd.DataFrame(rows)


def label_window(win_start_sec, win_end_sec, events_sec):
    label = 'Normal'
    for _, ev in events_sec.iterrows():
        overlap = max(0, min(win_end_sec, ev['end_sec']) -
                         max(win_start_sec, ev['start_sec']))
        if overlap / 30.0 > 0.5:
            label = ev['event']
            break
    return label


def create_windows(flow, thorac, spo2, df_events, recording_start):
    window_samples_32 = 960
    window_samples_4  = 120
    step_32           = 480
    step_4            = 60

    events_sec = df_events.copy()
    events_sec['start_sec'] = (df_events['start'] - recording_start).dt.total_seconds()
    events_sec['end_sec']   = (df_events['end']   - recording_start).dt.total_seconds()

    rows = []
    i    = 0
    while i + window_samples_32 <= len(flow):
        j = int(i * (window_samples_4 / window_samples_32))

        if j + window_samples_4 > len(spo2):
            break

        win_flow   = flow[i   : i + window_samples_32]
        win_thorac = thorac[i : i + window_samples_32]
        win_spo2   = spo2[j  : j + window_samples_4]

        win_start_sec = i / 32.0
        win_end_sec   = win_start_sec + 30.0

        label = label_window(win_start_sec, win_end_sec, events_sec)

        row = [win_start_sec, label]
        row.extend(win_flow.tolist())
        row.extend(win_thorac.tolist())
        row.extend(win_spo2.tolist())
        rows.append(row)

        i += step_32

    return rows

all_rows        = []
patient_folders = sorted(glob.glob(os.path.join(in_dir, 'AP*')))

print(f"Found {len(patient_folders)} patient folders\n")

for participant_path in patient_folders:
    participant_id = os.path.basename(participant_path)
    print(f"Processing {participant_id}...")

    events_path = os.path.join(participant_path, 'flow_events.txt')
    df_events   = load_events(events_path)

    times, flow_filt, thorac_filt, spo2_filt, df_flow, df_spo2 = load_patient_data(participant_path)

    recording_start = df_flow['time'].iloc[0]

    rows = create_windows(flow_filt, thorac_filt, spo2_filt, df_events, recording_start)

    for row in rows:
        row.insert(0, participant_id)

    all_rows.extend(rows)
    print(f"  {participant_id} → {len(rows)} windows created")

print(f"\nTotal windows across all patients: {len(all_rows)}")

flow_cols   = [f'f_{i}' for i in range(960)]
thorac_cols = [f't_{i}' for i in range(960)]
spo2_cols   = [f's_{i}' for i in range(120)]

columns = ['participant', 'window_start', 'label'] + flow_cols + thorac_cols + spo2_cols

df_dataset = pd.DataFrame(all_rows, columns=columns)

output_path = os.path.join(out_dir, 'breathing_dataset.csv')
df_dataset.to_csv(output_path, index=False)

print(f"Dataset saved to : {output_path}")
print(f"Shape            : {df_dataset.shape}")
print(f"\nLabel distribution:")
print(df_dataset['label'].value_counts())
print(f"\nParticipants:")
print(df_dataset['participant'].value_counts())

def load_sleep_profile(filepath):
    rows = []
    header_done = False
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                header_done = True
                continue
            if not header_done:
                continue
            parts = line.split(';')
            if len(parts) < 2:
                continue
            timestamp_str = parts[0].strip().replace(',', '.')
            stage         = parts[1].strip()
            try:
                ts = pd.to_datetime(timestamp_str, format='%d.%m.%Y %H:%M:%S.%f')
                rows.append({'time': ts, 'stage': stage})
            except Exception:
                continue
    return pd.DataFrame(rows)

def label_window_sleep(win_start_dt, df_sleep):
    idx = (df_sleep['time'] - win_start_dt).abs().argmin()
    return df_sleep.iloc[idx]['stage']


def create_sleep_windows(flow, thorac, spo2, df_sleep, recording_start):
    window_samples_32 = 960
    window_samples_4  = 120
    step_32           = 480

    rows = []
    i    = 0
    while i + window_samples_32 <= len(flow):
        j = int(i * (window_samples_4 / window_samples_32))
        if j + window_samples_4 > len(spo2):
            break

        win_flow   = flow[i   : i + window_samples_32]
        win_thorac = thorac[i : i + window_samples_32]
        win_spo2   = spo2[j  : j + window_samples_4]

        win_start_sec = i / 32.0
        win_start_dt  = recording_start + pd.Timedelta(seconds=win_start_sec)

        label = label_window_sleep(win_start_dt, df_sleep)

        row = [win_start_sec, label]
        row.extend(win_flow.tolist())
        row.extend(win_thorac.tolist())
        row.extend(win_spo2.tolist())
        rows.append(row)

        i += step_32

    return rows


all_sleep_rows  = []
patient_folders = sorted(glob.glob(os.path.join(in_dir, 'AP*')))

for participant_path in patient_folders:
    participant_id  = os.path.basename(participant_path)
    profile_path    = os.path.join(participant_path, 'sleep_profile.txt')
    df_sleep        = load_sleep_profile(profile_path)
    times, flow_filt, thorac_filt, spo2_filt, df_flow, df_spo2 = load_patient_data(participant_path)
    recording_start = df_flow['time'].iloc[0]
    rows            = create_sleep_windows(flow_filt, thorac_filt, spo2_filt, df_sleep, recording_start)
    for row in rows:
        row.insert(0, participant_id)
    all_sleep_rows.extend(rows)
    print(f"{participant_id} → {len(rows)} windows")

flow_cols   = [f'f_{i}' for i in range(960)]
thorac_cols = [f't_{i}' for i in range(960)]
spo2_cols   = [f's_{i}' for i in range(120)]
columns     = ['participant', 'window_start', 'label'] + flow_cols + thorac_cols + spo2_cols

df_sleep_dataset  = pd.DataFrame(all_sleep_rows, columns=columns)
sleep_output_path = os.path.join(out_dir, 'sleep_stage_dataset.csv')
df_sleep_dataset.to_csv(sleep_output_path, index=False)

print(f"\nSaved: {sleep_output_path}")
print(f"Shape: {df_sleep_dataset.shape}")
print(df_sleep_dataset['label'].value_counts())