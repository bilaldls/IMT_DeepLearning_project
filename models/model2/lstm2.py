#!/usr/bin/env python3
"""Train an LSTM to predict the next orbital elements from past TLEs.

Ce script :

- Utilise un split CHRONOLOGIQUE (train = début, val = milieu, test = fin).
- Utilise des features d'entrée (FEATURE_COLS) incluant les angles encodés en sin/cos.
- N'utilise comme cibles que les 7 éléments orbitaux "type papier":
    mean_motion_revs_per_day, eccentricity,
    inclination_deg, raan_deg, arg_perigee_deg, mean_anomaly_deg, bstar_drag
  mais numériquement, les angles sont encodés en sin/cos.
- Traite correctement l'epoch (epoch_unix ramené à 0 au début de la série).
- Purge les TLE aberrants avant l'entraînement.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# -----------------------------
# Définition des colonnes
# -----------------------------

# Angles (en degrés) présents dans le CSV
ANGLE_COLS_DEG = [
    "inclination_deg",
    "raan_deg",
    "arg_perigee_deg",
    "mean_anomaly_deg",
]

# 7 éléments orbitaux "type papier" utilisés comme CIBLE physique :
#   n (mean_motion), e, i, Ω, ω, M, BSTAR
ORBITAL_ELEM_BASE = [
    "mean_motion_revs_per_day",
    "eccentricity",
    "inclination_deg",
    "raan_deg",
    "arg_perigee_deg",
    "mean_anomaly_deg",
    "bstar_drag",
]

# Les angles seront encodés en sin/cos.
# On garde comme scalaires dans le label :
LABEL_SCALAR_COLS = [
    "mean_motion_revs_per_day",
    "eccentricity",
    "bstar_drag",
]


def angle_encoded_names() -> List[str]:
    """Retourne la liste des noms de colonnes sin/cos pour les angles."""
    encoded = []
    for col in ANGLE_COLS_DEG:
        base = col.replace("_deg", "")
        encoded.append(f"{base}_sin")
        encoded.append(f"{base}_cos")
    return encoded


ANGLE_ENCODED_COLS = angle_encoded_names()

# Colonnes de FEATURES (entrées du modèle) :
# On peut utiliser une base "physique" + les angles encodés.
# epoch_unix est utilisé comme feature, PAS comme label.
BASE_FEATURE_COLS = [
    "satellite_number",
    "intl_designator_launch_year",
    "intl_designator_launch_number",
    "epoch_year",
    "epoch_day",
    "first_derivative_mean_motion",
    "second_derivative_mean_motion",
    "bstar_drag",
    "ephemeris_type",
    "element_set_number",
    "eccentricity",
    "mean_motion_revs_per_day",
    "revolution_number_at_epoch",   # utilisé uniquement en entrée
    "epoch_unix",                   # temps depuis le début (en secondes)
]

FEATURE_COLS = BASE_FEATURE_COLS + ANGLE_ENCODED_COLS

# Colonnes de LABELS (sorties du modèle) :
#   -> uniquement les 7 éléments orbitaux type papier
#   -> angles encodés en sin/cos
LABEL_COLS = LABEL_SCALAR_COLS + ANGLE_ENCODED_COLS


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Chargement / preprocessing
# -----------------------------
def _add_epoch_unix(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne epoch_unix (secondes depuis epoch min)."""
    required_raw_cols = ["epoch", "satellite_number"]
    missing_raw = [c for c in required_raw_cols if c not in df.columns]
    if missing_raw:
        raise ValueError(f"Missing required raw columns in CSV: {', '.join(missing_raw)}")

    # Conversion de l'epoch texte → timestamp numérique (secondes Unix)
    try:
        epoch_dt = pd.to_datetime(df["epoch"], format="ISO8601")
    except (TypeError, ValueError):
        epoch_dt = pd.to_datetime(df["epoch"], format="mixed")

    epoch_unix = epoch_dt.astype("int64") // 10**9
    # On ramène l'epoch à 0 sur le minimum pour éviter des valeurs énormes
    epoch_unix = epoch_unix - epoch_unix.min()
    df["epoch_unix"] = epoch_unix.astype(np.float32)

    # Identifiant de satellite utilisé pour construire les séquences
    df["sat_id"] = df["satellite_number"].astype(int)

    return df


def _add_angle_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les colonnes d'angles en sin/cos. Les colonnes *_deg restent dans le df."""
    for col in ANGLE_COLS_DEG:
        if col not in df.columns:
            raise ValueError(f"Missing angle column in CSV: {col}")
        rad = np.deg2rad(df[col].astype(np.float32))
        base = col.replace("_deg", "")
        df[f"{base}_sin"] = np.sin(rad)
        df[f"{base}_cos"] = np.cos(rad)
    return df


def purge_aberrant_tles(df: pd.DataFrame) -> pd.DataFrame:
    """Purge les TLE aberrants (valeurs physiquement incohérentes ou extrêmes).

    Les bornes peuvent être adaptées, ici on met des valeurs raisonnables pour LEO/MEO/GTO.
    """
    # Vérification que les colonnes nécessaires existent
    needed = set(
        ANGLE_COLS_DEG
        + ORBITAL_ELEM_BASE
        + ["epoch_unix", "satellite_number"]
    )
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns before cleaning: {', '.join(missing)}")

    cond = (
        df["eccentricity"].between(0.0, 0.2)  # on coupe les cas très exotiques
        & df["inclination_deg"].between(0.0, 180.0)
        & df["mean_motion_revs_per_day"].between(0.5, 18.0)  # 0.5 à 18 rev/jour
        & df["bstar_drag"].between(-1e-3, 1e-3)  # bstar extrêmes virés
    )

    cleaned = df[cond].copy()

    # Supprimer les lignes avec NaN sur les colonnes critiques
    critical_cols = list(needed)
    cleaned = cleaned.dropna(subset=critical_cols)

    # Supprimer les doublons (même satellite, même epoch)
    cleaned = cleaned.drop_duplicates(subset=["satellite_number", "epoch_unix"])

    # Tri chronologique par satellite
    cleaned = cleaned.sort_values(["sat_id", "epoch_unix"]).reset_index(drop=True)
    return cleaned


def load_data(csv_path: str) -> pd.DataFrame:
    """Load TLE vectors from CSV, compute epoch_unix, encode angles, purge aberrants.

    Étapes :
    - charge le CSV
    - ajoute epoch_unix & sat_id
    - encode les angles en sin/cos
    - purge les TLE aberrants
    - vérifie que toutes les colonnes de FEATURES et LABELS existent
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    df = _add_epoch_unix(df)
    df = _add_angle_encodings(df)
    df = purge_aberrant_tles(df)

    # Vérification des colonnes de features et labels
    missing_feat = [c for c in FEATURE_COLS if c not in df.columns]
    missing_lab = [c for c in LABEL_COLS if c not in df.columns]
    if missing_feat:
        raise ValueError(f"Missing required feature columns: {', '.join(missing_feat)}")
    if missing_lab:
        raise ValueError(f"Missing required label columns: {', '.join(missing_lab)}")

    return df


# -----------------------------
# Scaling & séquences
# -----------------------------
def fit_and_scale(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Fit deux MinMaxScaler (features et labels) et retourne les arrays scalés."""
    feature_scaler = MinMaxScaler()
    label_scaler = MinMaxScaler()

    features = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    labels = df[LABEL_COLS].to_numpy(dtype=np.float32)

    features_scaled = feature_scaler.fit_transform(features)
    labels_scaled = label_scaler.fit_transform(labels)

    return features_scaled, labels_scaled, feature_scaler, label_scaler


def build_sequences(
    features_scaled: np.ndarray,
    labels_scaled: np.ndarray,
    sat_ids: Sequence,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding window sequences pour chaque satellite.

    X : (N, seq_len, feature_dim)
    y : (N, label_dim)   (vecteur de labels au pas suivant)
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    sat_ids = np.asarray(sat_ids)
    unique_sats = np.unique(sat_ids)

    for sat in unique_sats:
        idx = sat_ids == sat
        sat_feats = features_scaled[idx]
        sat_labs = labels_scaled[idx]

        if len(sat_feats) <= seq_len:
            continue

        for start in range(0, len(sat_feats) - seq_len):
            end = start + seq_len
            xs.append(sat_feats[start:end])
            ys.append(sat_labs[end])

    if not xs:
        raise ValueError("No sequences could be built; check seq_len and data size.")

    return np.stack(xs), np.stack(ys)


class TLEDataset(Dataset):
    """Torch Dataset wrapping TLE sequences and label targets."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.targets[idx]


# -----------------------------
# Modèle LSTM
# -----------------------------
class LSTMTLEModel(nn.Module):
    """LSTM followed by Linear head to predict next orbital elements vector."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # dernier pas de temps
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


# -----------------------------
# Split chrono & training
# -----------------------------
def split_datasets(
    sequences: np.ndarray,
    targets: np.ndarray,
    val_split: float,
    test_split: float,
) -> Tuple[TLEDataset, TLEDataset, TLEDataset]:
    """Split en train/val/test SANS shuffle (split chronologique)."""
    assert 0 < val_split < 1 and 0 < test_split < 1 and val_split + test_split < 1

    n = len(sequences)
    if n != len(targets):
        raise ValueError("sequences and targets must have the same length.")

    test_size = int(n * test_split)
    val_size = int(n * val_split)
    train_size = n - val_size - test_size

    if train_size <= 0:
        raise ValueError("Train split is empty. Check val_split and test_split values.")

    train_seq = sequences[:train_size]
    val_seq = sequences[train_size : train_size + val_size]
    test_seq = sequences[train_size + val_size :]

    train_tgt = targets[:train_size]
    val_tgt = targets[train_size : train_size + val_size]
    test_tgt = targets[train_size + val_size :]

    return (
        TLEDataset(train_seq, train_tgt),
        TLEDataset(val_seq, val_tgt),
        TLEDataset(test_seq, test_tgt),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute MAE on a DataLoader."""
    criterion = nn.L1Loss()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
) -> Tuple[nn.Module, List[float], List[float]]:
    """Train model with Adam, ReduceLROnPlateau, MAE loss + early stopping."""
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    model.to(device)
    train_history: List[float] = []
    val_history: List[float] = []
    best_val = float("inf")
    best_state = None
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            epoch_loss += loss.item() * batch_size
            total += batch_size

        train_mae = epoch_loss / max(total, 1)
        val_mae = evaluate(model, val_loader, device)
        scheduler.step(val_mae)

        train_history.append(train_mae)
        val_history.append(val_mae)

        print(f"Epoch {epoch:03d} | train MAE={train_mae:.6f} | val MAE={val_mae:.6f}")

        if val_mae < best_val:
            best_val = val_mae
            best_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(no val MAE improvement for {patience} consecutive epochs)."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_history, val_history


# -----------------------------
# Arguments & main
# -----------------------------
@dataclass
class TrainArgs:
    csv: str
    seq_len: int
    hidden: int
    layers: int
    dropout: float
    batch_size: int
    epochs: int
    lr: float
    val_split: float
    test_split: float
    seed: int
    model_path: str
    scaler_path: str
    early_stopping_patience: int


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train LSTM to predict next orbital elements vector.")
    parser.add_argument(
        "--csv",
        default="/Users/bilaldelais/Desktop/project deep learning/data/processed/iss_200000_parsed.csv",
        help="Path to CSV containing parsed TLEs.",
    )
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length L.")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size of LSTM.")
    parser.add_argument("--layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Number of epochs with no val MAE improvement before early stopping.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-split", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--model-path", default="lstm_tle_model.pth", help="Path to save trained model.")
    parser.add_argument(
        "--scaler-path",
        default="scalers.pkl",
        help="Path to save fitted feature & label scalers.",
    )

    args_ns = parser.parse_args()

    return TrainArgs(
        csv=args_ns.csv,
        seq_len=args_ns.seq_len,
        hidden=args_ns.hidden,
        layers=args_ns.layers,
        dropout=args_ns.dropout,
        batch_size=args_ns.batch_size,
        epochs=args_ns.epochs,
        lr=args_ns.lr,
        val_split=args_ns.val_split,
        test_split=args_ns.test_split,
        seed=args_ns.seed,
        model_path=args_ns.model_path,
        scaler_path=args_ns.scaler_path,
        early_stopping_patience=args_ns.early_stopping_patience,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(f"Device: {device}")

    df = load_data(args.csv)

    # Corrélation sur les features d'entrée
    corr = df[FEATURE_COLS].corr()
    print("Feature correlation matrix:")
    print(corr)
    corr.to_csv("feature_correlation_matrix.csv")
    print("Correlation matrix saved to feature_correlation_matrix.csv")

    # Scaling features & labels
    features_scaled, labels_scaled, feature_scaler, label_scaler = fit_and_scale(df)

    # Séquences + split chrono
    sequences, targets = build_sequences(
        features_scaled, labels_scaled, df["sat_id"].to_numpy(), args.seq_len
    )

    train_ds, val_ds, test_ds = split_datasets(
        sequences,
        targets,
        val_split=args.val_split,
        test_split=args.test_split,
    )
    print(
        f"Dataset sizes | train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = LSTMTLEModel(
        input_size=sequences.shape[-1],
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        output_size=targets.shape[-1],
    )

    model, train_hist, val_hist = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.early_stopping_patience,
    )

    test_mae = evaluate(model, test_loader, device)
    print(f"Test MAE (sur les 7 éléments orbitaux encodés) : {test_mae:.6f}")

    # Log structuré
    log_df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(train_hist) + 1),
            "train_mae": train_hist,
            "val_mae": val_hist,
        }
    )
    log_df["test_mae"] = np.nan
    log_df.loc[len(log_df) - 1, "test_mae"] = test_mae
    log_df.to_csv("training_log.csv", index=False)
    print("Training log saved to training_log.csv")

    # Courbe train/val
    plt.figure()
    epochs_arr = np.arange(1, len(train_hist) + 1)
    plt.plot(epochs_arr, train_hist, label="train MAE")
    plt.plot(epochs_arr, val_hist, label="val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (éléments orbitaux encodés)")
    plt.title("Train vs Validation MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_val_mae.png")
    plt.close()
    print("Train/val MAE curve saved to train_val_mae.png")

    # Sauvegarde modèle + scalers (features + labels)
    torch.save(model.state_dict(), args.model_path)
    with open(args.scaler_path, "wb") as f:
        pickle.dump(
            {
                "feature_scaler": feature_scaler,
                "label_scaler": label_scaler,
                "feature_cols": FEATURE_COLS,
                "label_cols": LABEL_COLS,
            },
            f,
        )

    print(f"Model saved to {args.model_path}")
    print(f"Scalers saved to {args.scaler_path}")


if __name__ == "__main__":
    main()