import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

participant_path = "../data/AP01"
participant_id   = os.path.basename(participant_path)

def find_file(folder, keyword):
    for fname in os.listdir(folder):
        if keyword.lower() in fname.lower() and fname.endswith('.txt'):
            return os.path.join(folder, fname)
    raise FileNotFoundError(f"No file with '{keyword}' found in {folder}")
flow_path    = find_file(participant_path, 'nasal_airflow')
thorac_path  = find_file(participant_path, 'thoracic_movement')
spo2_path    = find_file(participant_path, 'spo2')
events_path  = find_file(participant_path, 'flow_events')
profile_path = find_file(participant_path, 'sleep_profile')
print(flow_path)
print(thorac_path)
print(spo2_path)
print(events_path)

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

df_flow   = load_signal(flow_path)
df_thorac = load_signal(thorac_path)
df_spo2   = load_signal(spo2_path)

print(df_flow.shape, df_thorac.shape, df_spo2.shape)
print(df_flow.head(3))

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
                end_dt    = pd.to_datetime(date_part + ' ' + end_str, format='%d.%m.%Y %H:%M:%S.%f')
                if end_dt < start_dt:
                    end_dt += pd.Timedelta(days=1)
                rows.append({'start': start_dt, 'end': end_dt, 'event': event_type})
            except Exception:
                continue

    return pd.DataFrame(rows)

df_events = load_events(events_path)

print(df_events.shape)
print(df_events['event'].value_counts())
print(df_events.head(3))

EVENT_COLORS = {
    'Hypopnea'         : ('orange', 0.35),
    'Obstructive Apnea': ('red',    0.35),
    'Central Apnea'    : ('purple', 0.35),
    'Mixed Apnea'      : ('brown',  0.35),
}

def get_event_color(event_name):
    for key, val in EVENT_COLORS.items():
        if key.lower() in event_name.lower():
            return val
    return ('gray', 0.3)

def draw_events(ax, df_events, t_start, t_end):
    window_events = df_events[
        (df_events['end']   > t_start) &
        (df_events['start'] < t_end)
    ]
    for _, ev in window_events.iterrows():
        color, alpha = get_event_color(ev['event'])
        ev_start = max(ev['start'], t_start)
        ev_end   = min(ev['end'],   t_end)
        ax.axvspan(ev_start, ev_end, color=color, alpha=alpha, zorder=2)

os.makedirs('../Visualizations', exist_ok=True)
output_pdf = os.path.join('../Visualizations', f'{participant_id}_visualization.pdf')

t_global_start = df_flow['time'].min()
t_global_end   = df_flow['time'].max()
chunk_delta    = pd.Timedelta(minutes=5)

chunks = []
t = t_global_start
while t < t_global_end:
    chunks.append((t, min(t + chunk_delta, t_global_end)))
    t += chunk_delta

legend_patches = [
    mpatches.Patch(color='red',    alpha=0.5, label='Obstructive Apnea'),
    mpatches.Patch(color='orange', alpha=0.5, label='Hypopnea'),
    mpatches.Patch(color='purple', alpha=0.5, label='Central Apnea'),
]

with PdfPages(output_pdf) as pdf:
    for i, (t_start, t_end) in enumerate(chunks):

        mask_flow   = (df_flow['time']   >= t_start) & (df_flow['time']   <= t_end)
        mask_thorac = (df_thorac['time'] >= t_start) & (df_thorac['time'] <= t_end)
        mask_spo2   = (df_spo2['time']   >= t_start) & (df_spo2['time']   <= t_end)

        chunk_flow   = df_flow[mask_flow]
        chunk_thorac = df_thorac[mask_thorac]
        chunk_spo2   = df_spo2[mask_spo2]

        fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True,
                                 gridspec_kw={'hspace': 0.35})

        title = (f"{participant_id}  |  "
                 f"{t_start.strftime('%d-%m-%Y  %H:%M:%S')}  to  "
                 f"{t_end.strftime('%H:%M:%S')}")
        fig.suptitle(title, fontsize=11, fontweight='bold', y=0.98)

        axes[0].plot(chunk_flow['time'], chunk_flow['value'],
                     color='steelblue', linewidth=0.6, label='Nasal Flow')
        axes[0].set_ylabel('Nasal Flow\n(a.u.)', fontsize=8)
        axes[0].legend(loc='upper right', fontsize=7)
        axes[0].tick_params(axis='both', labelsize=7)
        axes[0].grid(True, alpha=0.3)
        draw_events(axes[0], df_events, t_start, t_end)

        axes[1].plot(chunk_thorac['time'], chunk_thorac['value'],
                     color='darkorange', linewidth=0.6, label='Thoracic/Abdominal Resp.')
        axes[1].set_ylabel('Thoracic Resp.\n(a.u.)', fontsize=8)
        axes[1].legend(loc='upper right', fontsize=7)
        axes[1].tick_params(axis='both', labelsize=7)
        axes[1].grid(True, alpha=0.3)
        draw_events(axes[1], df_events, t_start, t_end)

        axes[2].plot(chunk_spo2['time'], chunk_spo2['value'],
                     color='dimgray', linewidth=0.8, label='SpO₂')
        axes[2].set_ylabel('SpO₂\n(%)', fontsize=8)
        axes[2].set_ylim(
            max(0,   chunk_spo2['value'].min() - 2) if len(chunk_spo2) > 0 else 85,
            min(100, chunk_spo2['value'].max() + 2) if len(chunk_spo2) > 0 else 100
        )
        axes[2].legend(loc='upper right', fontsize=7)
        axes[2].tick_params(axis='both', labelsize=7)
        axes[2].grid(True, alpha=0.3)
        draw_events(axes[2], df_events, t_start, t_end)

        axes[2].set_xlabel('Time', fontsize=8)
        fig.autofmt_xdate(rotation=45, ha='right')

        fig.legend(handles=legend_patches, loc='lower center',
                   ncol=3, fontsize=8, framealpha=0.8,
                   bbox_to_anchor=(0.5, 0.01))

        pdf.savefig(fig, bbox_inches='tight', dpi=100)
        plt.close(fig)

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"Page {i+1} / {len(chunks)} done")

print(f"Saved: {output_pdf}")