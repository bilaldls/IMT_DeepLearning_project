import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class SGP4CorrectionDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class SGP4CorrectionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_dim, 3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out

# -----------------------------
# EXEMPLE DE CHARGEMENT
# -----------------------------

# Lecture des données réelles depuis le fichier CSV
# (adapter le chemin si besoin, par ex. "./data/TERRA.csv")
df = pd.read_csv("TERRA.csv")

# Conversion du temps en secondes depuis le début de la série
df["epoch_utc"] = pd.to_datetime(df["epoch_utc"])
t0 = df["epoch_utc"].min()
df["dt_sec"] = (df["epoch_utc"] - t0).dt.total_seconds()

# Colonnes de features
# TLE / éléments orbitaux moyens
tle_cols = [
    "incl_deg",
    "raan_deg",
    "ecc",
    "argp_deg",
    "M_deg",
    "mean_motion_rev_per_day",
]

# Sortie SGP4 : position + vitesse (dans le repère TEME ici)
sgp4_cols = [
    "sgp4_x_km",
    "sgp4_y_km",
    "sgp4_z_km",
    "sgp4_vx_km_s",
    "sgp4_vy_km_s",
    "sgp4_vz_km_s",
]

# Feature de temps relatif
time_cols = ["dt_sec"]

feature_cols = sgp4_cols + time_cols + tle_cols

# Matrice d'entrée X_all : toutes les lignes, toutes les features utiles
X_all = df[feature_cols].values.astype(np.float32)

# Cible : correction de la position vraie par rapport à SGP4
true_pos = df[["true_x_km", "true_y_km", "true_z_km"]].values.astype(np.float32)
sgp4_pos = df[["sgp4_x_km", "sgp4_y_km", "sgp4_z_km"]].values.astype(np.float32)
Y_all = true_pos - sgp4_pos

# Construction de séquences glissantes pour alimenter le LSTM
# Chaque séquence contient "seq_len" instants consécutifs
seq_len = 50  # tu peux ajuster cette valeur

def build_sequences(X, Y, seq_len):
    N = X.shape[0]
    if N < seq_len:
        raise ValueError(f"Nombre d'echantillons ({N}) < seq_len ({seq_len})")
    N_seq = N - seq_len + 1
    X_seq = np.zeros((N_seq, seq_len, X.shape[1]), dtype=np.float32)
    Y_seq = np.zeros((N_seq, seq_len, Y.shape[1]), dtype=np.float32)
    for i in range(N_seq):
        X_seq[i] = X[i : i + seq_len]
        Y_seq[i] = Y[i : i + seq_len]
    return X_seq, Y_seq

X, Y = build_sequences(X_all, Y_all, seq_len)
N_seq = X.shape[0]

input_dim = X.shape[2]

split = int(0.8 * N_seq)
X_train, Y_train = X[:split], Y[:split]
X_val,   Y_val   = X[split:], Y[split:]

train_ds = SGP4CorrectionDataset(X_train, Y_train)
val_ds   = SGP4CorrectionDataset(X_val,   Y_val)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SGP4CorrectionLSTM(input_dim=input_dim)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 30
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
    train_loss /= len(train_ds)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)
    val_loss /= len(val_ds)
    print(f"Epoch {epoch+1}/{n_epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")