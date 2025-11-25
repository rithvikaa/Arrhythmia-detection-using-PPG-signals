import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Load dataset
df = pd.read_csv(r'C:\Users\Admin\Downloads\archu\archive (1)\PPG_Dataset.csv')

# Prepare features and labels
X = df.iloc[:, :-1].values
y = df['Label'].values

# Encode labels to integers
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Preprocessing helper functions
def bandpass_filter(signal, lowcut=0.5, highcut=8.0, fs=100, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def preprocess_signal(signal):
    smooth = savgol_filter(signal, window_length=51, polyorder=3)
    filtered = bandpass_filter(smooth)
    return filtered

# Apply preprocessing
X_processed = np.array([preprocess_signal(sample) for sample in X])
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_processed)

# Reshape for PyTorch: (samples, channels=1, timesteps=2000)
X_norm = np.expand_dims(X_norm, axis=1).astype(np.float32)
y_enc = y_enc.astype(np.int64)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

# Dataset class definition
class PPGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = PPGDataset(X_train, y_train)
test_dataset = PPGDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Model architecture: ResNet, BiGRU, Attention
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        weights = self.attention(x).squeeze(-1)
        weights = torch.softmax(weights, dim=1).unsqueeze(-1)
        weighted = x * weights
        out = torch.sum(weighted, dim=1)
        return out, weights

class ArrhythmiaModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.resblock1 = ResNetBlock(64)
        self.resblock2 = ResNetBlock(64)
        self.gru = nn.GRU(64, 64, bidirectional=True, batch_first=True)
        self.attention = Attention(128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = x.permute(0, 2, 1)  # to (batch, seq_len, features)
        gru_out, _ = self.gru(x)
        attn_out, _ = self.attention(gru_out)
        output = self.fc(attn_out)
        return output

# Initialize device, model, loss and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ArrhythmiaModel(num_classes=len(np.unique(y_enc))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with progress bar
best_acc = 0
for epoch in range(2):
    model.train()
    total_correct, total_loss = 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()
    train_acc = total_correct / len(train_dataset)
    train_loss = total_loss / len(train_dataset)

    # Validation
    model.eval()
    total_correct, total_loss = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
    val_acc = total_correct / len(test_dataset)
    val_loss = total_loss / len(test_dataset)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_arrhythmia_model.pth')

print(f"Best Validation Accuracy: {best_acc:.4f}")

# Load best model for final evaluation and individual predictions
model.load_state_dict(torch.load('best_arrhythmia_model.pth'))
model.eval()

# Predict and show arrhythmia detection for each test sample
print("\nArrhythmia detection results on test samples:")
with torch.no_grad():
    for i in range(len(test_dataset)):
        sample, label = test_dataset[i]
        sample = sample.unsqueeze(0).to(device)  # batch dimension
        output = model(sample)
        pred = torch.argmax(output, dim=1).item()
        true_label = le.inverse_transform([label])[0]
        pred_label = le.inverse_transform([pred])[0]
        print(f"Sample {i+1}: True label = {true_label}, Predicted = {pred_label} --> {'Arrhythmia Detected' if pred_label == 'MI' else 'No Arrhythmia'}")
