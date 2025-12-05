#!/usr/bin/env python3
"""Train an LSTM to predict the next TLE vector from sequences of past TLEs.

This version:
- Utilise un split CHRONOLOGIQUE (train = début, val = milieu, test = fin).
- Utilise comme features toutes les colonnes numériques de ton dataset TLE
  (plus une colonne dérivée epoch_unix à partir de la colonne epoch texte).
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

# Toutes les colonnes FEATURES issues de ton CSV + epoch_unix dérivé de "epoch".
# On exclut uniquement les colonnes textuelles (epoch, name, classification,
# intl_designator_piece) qui ne sont pas directement numériques.
FEATURE_COLS = [
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
    "inclination_deg",
    "raan_deg",
    "eccentricity",
    "arg_perigee_deg",
    "mean_anomaly_deg",
    "mean_motion_revs_per_day",
    "revolution_number_at_epoch",
    "epoch_unix",  # ajouté à partir de la colonne texte "epoch"
]


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load TLE vectors from CSV and ensure required columns exist.

    Modifications importantes :
    ---------------------------
    - On part de ton CSV (avec colonnes: epoch, name, satellite_number, etc.).
    - On crée deux nouvelles colonnes internes :
        * epoch_unix : epoch converti en secondes (float)
        * sat_id     : identifiant numérique du satellite (= satellite_number)
    - On vérifie que toutes les colonnes de FEATURE_COLS existent.
    - On trie par (sat_id, epoch_unix) pour garantir l'ordre temporel.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Colonnes minimales attendues dans le CSV brut
    required_raw_cols = ["epoch", "satellite_number"]
    missing_raw = [c for c in required_raw_cols if c not in df.columns]
    if missing_raw:
        raise ValueError(f"Missing required raw columns in CSV: {', '.join(missing_raw)}")

    # Conversion de l'epoch texte → timestamp numérique (secondes Unix)
    # On supporte les formats ISO8601 hétérogènes (avec/sans microsecondes, avec décalage horaire).
    try:
        epoch_dt = pd.to_datetime(df["epoch"], format="ISO8601")
    except (TypeError, ValueError):
        # Fallback au mode 'mixed' si jamais certains formats sont exotiques
        epoch_dt = pd.to_datetime(df["epoch"], format="mixed")
    df["epoch_unix"] = epoch_dt.astype("int64") // 10**9

    # Identifiant de satellite utilisé pour construire les séquences
    df["sat_id"] = df["satellite_number"]

    # Vérification que toutes les features demandées existent maintenant
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {', '.join(missing_features)}")

    # Tri chronologique par satellite (fondamental pour le split chrono)
    df = df.sort_values(["sat_id", "epoch_unix"]).reset_index(drop=True)
    return df


def fit_and_scale(df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    """Fit MinMaxScaler on feature columns and return scaled numpy array."""
    scaler = MinMaxScaler()
    # On ne scale que les colonnes de FEATURE_COLS (toutes numériques)
    scaled = scaler.fit_transform(df[FEATURE_COLS].to_numpy(dtype=np.float32))
    return scaled, scaler


def build_sequences(
    features_scaled: np.ndarray, sat_ids: Sequence, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding window sequences for each satellite.

    - On suppose que features_scaled est déjà trié par (sat_id, epoch_unix).
    - Pour chaque satellite, on crée des fenêtres glissantes de longueur seq_len
      et on demande au modèle de prédire le vecteur suivant.

    Returns
    -------
    X : np.ndarray of shape (N, seq_len, feature_dim)
    y : np.ndarray of shape (N, feature_dim)
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    sat_ids = np.asarray(sat_ids)
    unique_sats = np.unique(sat_ids)

    for sat in unique_sats:
        idx = sat_ids == sat
        sat_feats = features_scaled[idx]
        if len(sat_feats) <= seq_len:
            continue
        for start in range(0, len(sat_feats) - seq_len):
            end = start + seq_len
            xs.append(sat_feats[start:end])
            ys.append(sat_feats[end])

    if not xs:
        raise ValueError("No sequences could be built; check seq_len and data size.")

    return np.stack(xs), np.stack(ys)


class TLEDataset(Dataset):
    """Torch Dataset wrapping TLE sequences and targets."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.targets[idx]


class LSTMTLEModel(nn.Module):
    """LSTM followed by Linear head to predict next TLE vector."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        # PyTorch n'applique le dropout entre les couches LSTM que si num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # On récupère le dernier pas de temps de la séquence
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


def split_datasets(
    sequences: np.ndarray,
    targets: np.ndarray,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[TLEDataset, TLEDataset, TLEDataset]:
    """Split into train/val/test datasets WITHOUT shuffling (chronological split).

    Changement principal par rapport à une version avec mélange aléatoire :
    - AUCUN shuffle n'est appliqué sur les séquences.
    - On coupe simplement le tableau dans l'ordre :
        * [0 : train_end]       → train (début de la série)
        * [train_end : val_end] → validation (milieu)
        * [val_end : n]         → test (fin de la série)

    Le paramètre `seed` est conservé pour compatibilité, mais il n'est pas
    utilisé dans le split lui-même.
    """
    assert 0 < val_split < 1 and 0 < test_split < 1 and val_split + test_split < 1

    n = len(sequences)
    if n != len(targets):
        raise ValueError("sequences and targets must have the same length.")

    test_size = int(n * test_split)
    val_size = int(n * val_split)
    train_size = n - val_size - test_size

    if train_size <= 0:
        raise ValueError("Train split is empty. Check val_split and test_split values.")

    # Split chronologique
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
) -> Tuple[nn.Module, List[float], List[float]]:
    """Train model with Adam, ReduceLROnPlateau, MAE loss."""
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

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_history, val_history


def predict_future_tles(
    model: nn.Module,
    scaler: MinMaxScaler,
    tle_sequence: np.ndarray,
    horizon_k: int,
    device: torch.device | None = None,
) -> np.ndarray:
    """Iteratively predict k future TLE vectors given the last seq_len vectors.

    Parameters
    ----------
    model : trained LSTMTLEModel
    scaler : fitted MinMaxScaler
    tle_sequence : np.ndarray, shape (seq_len, feature_dim) in original scale
    horizon_k : int, number of future steps to predict
    device : torch.device or None

    Returns
    -------
    np.ndarray of shape (horizon_k, feature_dim) in original scale.
    """
    model.eval()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # On scale la séquence d'entrée avec le même scaler que lors de l'entraînement
    seq = scaler.transform(tle_sequence.astype(np.float32))
    preds: List[np.ndarray] = []

    with torch.no_grad():
        for _ in range(horizon_k):
            # On utilise la dernière fenêtre de même longueur que la séquence initiale
            x = torch.tensor(seq[-len(tle_sequence) :][None, ...], dtype=torch.float32, device=device)
            out = model(x)
            next_scaled = out.squeeze(0).cpu().numpy()
            preds.append(next_scaled)
            # On ajoute la prédiction à la suite pour prédire encore plus loin
            seq = np.vstack([seq, next_scaled])

    preds_array = np.vstack(preds)
    return scaler.inverse_transform(preds_array)


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


def parse_args() -> TrainArgs:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train LSTM to predict next TLE vector.")
    parser.add_argument(
        "--csv",
        default="/Users/bilaldelais/Desktop/project deep learning/data/processed/iss_200000_parsed.csv",
        help="Path to CSV containing TLE vectors."
    )
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length L.")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size of LSTM.")
    parser.add_argument("--layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-split", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--model-path", default="lstm_tle_model.pth", help="Path to save trained model.")
    parser.add_argument("--scaler-path", default="scaler.pkl", help="Path to save fitted scaler.")

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
    )


def main() -> None:
    args = parse_args()

    # Force le chemin automatiquement
    #args.csv = "/Users/bilaldelais/Desktop/project deep learning/data/iss_200000_parsed.csv"
    
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
    features_scaled, scaler = fit_and_scale(df)
    sequences, targets = build_sequences(features_scaled, df["sat_id"].to_numpy(), args.seq_len)

    # Split chronologique (pas de shuffle avant découpage)
    train_ds, val_ds, test_ds = split_datasets(
        sequences,
        targets,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    print(
        f"Dataset sizes | train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}"
    )

    # On peut mélanger les batches à l'intérieur du train_loader sans casser
    # la cohérence temporelle globale du split.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = LSTMTLEModel(
        input_size=sequences.shape[-1],
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    )

    model, train_hist, val_hist = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )

    test_mae = evaluate(model, test_loader, device)
    print(f"Test MAE: {test_mae:.6f}")

    torch.save(model.state_dict(), args.model_path)
    with open(args.scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Model saved to {args.model_path}")
    print(f"Scaler saved to {args.scaler_path}")


if __name__ == "__main__":
    main()
