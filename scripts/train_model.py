import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset_path = "../Dataset/breathing_dataset.csv"
df           = pd.read_csv(dataset_path)

flow_cols   = [f'f_{i}' for i in range(960)]
thorac_cols = [f't_{i}' for i in range(960)]
spo2_cols   = [f's_{i}' for i in range(120)]

feature_cols = flow_cols + thorac_cols + spo2_cols

X            = df[feature_cols].values
y            = df['label'].values
participants = df['participant'].values

print(f"Dataset shape     : {df.shape}")
print(f"Features shape    : {X.shape}")
print(f"Unique labels     : {np.unique(y)}")
print(f"Unique patients   : {np.unique(participants)}")
print(f"\nLabel distribution:")
print(pd.Series(y).value_counts())

le      = LabelEncoder()
y_enc   = le.fit_transform(y)

print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

flow_data   = X[:, :960]
thorac_data = X[:, 960:1920]
spo2_data   = X[:, 1920:]

spo2_upsampled = np.repeat(spo2_data, 8, axis=1)

X_3ch = np.stack([flow_data, thorac_data, spo2_upsampled], axis=1)

print(f"Input shape for CNN: {X_3ch.shape}")

class CNN1D(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN1D, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x
    
class BreathingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, train_loader, num_epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:2d}/{num_epochs} — Loss: {avg_loss:.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch  = X_batch.to(device)
            outputs  = model(X_batch)
            preds    = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec  = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds)

    return acc, prec, rec, cm, all_preds, all_labels

from sklearn.preprocessing import StandardScaler

unique_participants = np.unique(participants)

all_true  = []
all_pred  = []
fold_results = []

print("Starting Leave-One-Participant-Out Cross Validation\n")
print("=" * 55)

for test_participant in unique_participants:
    print(f"\nFold: Test on {test_participant}")

    train_mask = participants != test_participant
    test_mask  = participants == test_participant

    X_train = X_3ch[train_mask]
    y_train = y_enc[train_mask]
    X_test  = X_3ch[test_mask]
    y_test  = y_enc[test_mask]

    scaler       = StandardScaler()
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat  = X_test.reshape(len(X_test), -1)

    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat  = scaler.transform(X_test_flat)

    X_train = X_train_flat.reshape(len(X_train), 3, 960)
    X_test  = X_test_flat.reshape(len(X_test), 3, 960)

    train_dataset = BreathingDataset(X_train, y_train)
    test_dataset  = BreathingDataset(X_test,  y_test)

    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    model = CNN1D(num_classes=len(le.classes_))
    train_model(model, train_loader, num_epochs=30)

    acc, prec, rec, cm, preds, labels = evaluate_model(model, test_loader)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")

    fold_results.append({'participant': test_participant,
                         'accuracy': acc,
                         'precision': prec,
                         'recall': rec})

    all_true.extend(labels)
    all_pred.extend(preds)

print("\n" + "=" * 55)

all_true = np.array(all_true)
all_pred = np.array(all_pred)

final_acc  = accuracy_score(all_true, all_pred)
final_prec = precision_score(all_true, all_pred, average='weighted', zero_division=0)
final_rec  = recall_score(all_true, all_pred, average='weighted', zero_division=0)
final_cm   = confusion_matrix(all_true, all_pred)

print("\nFINAL RESULTS ACROSS ALL FOLDS")
print("=" * 55)
print(f"Overall Accuracy  : {final_acc:.4f}")
print(f"Overall Precision : {final_prec:.4f}")
print(f"Overall Recall    : {final_rec:.4f}")

print("\nPer-Fold Results:")
for fold in fold_results:
    print(f"  {fold['participant']} → "
          f"Acc: {fold['accuracy']:.4f}  "
          f"Prec: {fold['precision']:.4f}  "
          f"Rec: {fold['recall']:.4f}")

print("\nConfusion Matrix:")
print(f"Classes: {le.classes_}")
print(final_cm)

print("\nDetailed Classification Report:")
print(classification_report(all_true, all_pred,
                             target_names=le.classes_,
                             zero_division=0))

results_df = pd.DataFrame(fold_results)
results_df.to_csv('../Dataset/lopo_results.csv', index=False)
print("\nFold results saved to ../Dataset/lopo_results.csv")