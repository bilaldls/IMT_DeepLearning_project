#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Script d'évaluation (v2) : compare les TLE prédits par le nouveau LSTM
# avec les TLE réels en propageant les deux via SGP4 (erreur de position en km).
#
# Adapté au modèle :
#  - features = colonnes physiques + angles encodés en sin/cos
#  - labels   = 7 éléments orbitaux "type papier" :
#        mean_motion_revs_per_day, eccentricity, bstar_drag
#        + (inclination, raan, arg_perigee, mean_anomaly) encodés en sin/cos
#  - 2 scalers : feature_scaler et label_scaler, stockés dans scalers.pkl
#  - Ajout :
#        * tests de cohérence SGP4,
#        * erreur à t0,
#        * erreurs sur horizon complet et horizon court (~90 min).
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import matplotlib.pyplot as plt

from sgp4.api import Satrec, WGS72, jday

# -----------------------------------------------------------------------------
# Colonnes et constantes communes (alignées avec le script d'entraînement v2)
# -----------------------------------------------------------------------------

# Angles présents dans le CSV, en degrés
ANGLE_COLS_DEG = [
    "inclination_deg",
    "raan_deg",
    "arg_perigee_deg",
    "mean_anomaly_deg",
]


def angle_encoded_names() -> List[str]:
    encoded = []
    for col in ANGLE_COLS_DEG:
        base = col.replace("_deg", "")
        encoded.append(f"{base}_sin")
        encoded.append(f"{base}_cos")
    return encoded


ANGLE_ENCODED_COLS = angle_encoded_names()

# Les scalers fourniront FEATURE_COLS et LABEL_COLS au runtime
FEATURE_COLS: List[str] = []
LABEL_COLS: List[str] = []

# Origine SGP4 (jours depuis 1949-12-31)
EPOCH0 = datetime(1949, 12, 31, tzinfo=timezone.utc)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Chargement / preprocessing des données (identique au train v2)
# -----------------------------------------------------------------------------
def _add_epoch_unix(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute epoch_unix (secondes depuis le min) et sat_id."""
    required_raw_cols = ["epoch", "satellite_number"]
    missing_raw = [c for c in required_raw_cols if c not in df.columns]
    if missing_raw:
        raise ValueError(f"Missing required raw columns in CSV: {', '.join(missing_raw)}")

    try:
        epoch_dt = pd.to_datetime(df["epoch"], format="ISO8601")
    except (TypeError, ValueError):
        epoch_dt = pd.to_datetime(df["epoch"], format="mixed")

    epoch_unix = epoch_dt.astype("int64") // 10**9
    epoch_unix = epoch_unix - epoch_unix.min()
    df["epoch_unix"] = epoch_unix.astype(np.float32)
    df["sat_id"] = df["satellite_number"].astype(int)
    return df


def _add_angle_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les colonnes d'angles en sin/cos. Les *_deg restent dans le df."""
    for col in ANGLE_COLS_DEG:
        if col not in df.columns:
            raise ValueError(f"Missing angle column in CSV: {col}")
        rad = np.deg2rad(df[col].astype(np.float32))
        base = col.replace("_deg", "")
        df[f"{base}_sin"] = np.sin(rad)
        df[f"{base}_cos"] = np.cos(rad)
    return df


def purge_aberrant_tles(df: pd.DataFrame) -> pd.DataFrame:
    """Purge les TLE aberrants selon des bornes raisonnables."""
    needed = set(
        ANGLE_COLS_DEG
        + [
            "mean_motion_revs_per_day",
            "eccentricity",
            "bstar_drag",
            "epoch_unix",
            "satellite_number",
        ]
    )
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns before cleaning: {', '.join(missing)}")

    cond = (
        df["eccentricity"].between(0.0, 0.2)
        & df["inclination_deg"].between(0.0, 180.0)
        & df["mean_motion_revs_per_day"].between(0.5, 18.0)
        & df["bstar_drag"].between(-1e-3, 1e-3)
    )

    cleaned = df[cond].copy()
    critical_cols = list(needed)
    cleaned = cleaned.dropna(subset=critical_cols)
    cleaned = cleaned.drop_duplicates(subset=["satellite_number", "epoch_unix"])
    cleaned = cleaned.sort_values(["sat_id", "epoch_unix"]).reset_index(drop=True)
    return cleaned


def load_data(csv_path: str) -> pd.DataFrame:
    """Charge le CSV TLE, ajoute epoch_unix, sin/cos, purge, trie."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _add_epoch_unix(df)
    df = _add_angle_encodings(df)
    df = purge_aberrant_tles(df)
    return df


# -----------------------------------------------------------------------------
# Séquences (features + labels) avec indices des TLE cibles
# -----------------------------------------------------------------------------
def build_sequences_with_indices(
    features_scaled: np.ndarray,
    labels_scaled: np.ndarray,
    sat_ids: Sequence,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit des séquences glissantes par satellite, en renvoyant:

    X : (N, seq_len, D_feat)  séquences de features
    y : (N, D_label)          labels du pas suivant
    idx_target : (N,)         indices du TLE cible dans le DataFrame original
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    idxs: List[int] = []

    sat_ids = np.asarray(sat_ids)
    all_indices = np.arange(len(sat_ids))
    unique_sats = np.unique(sat_ids)

    for sat in unique_sats:
        mask = sat_ids == sat
        sat_feats = features_scaled[mask]
        sat_labs = labels_scaled[mask]
        sat_idx = all_indices[mask]

        if len(sat_feats) <= seq_len:
            continue

        for start in range(0, len(sat_feats) - seq_len):
            end = start + seq_len
            xs.append(sat_feats[start:end])
            ys.append(sat_labs[end])
            idxs.append(int(sat_idx[end]))

    if not xs:
        raise ValueError("No sequences could be built; check seq_len and data size.")

    return np.stack(xs), np.stack(ys), np.array(idxs, dtype=int)


def chronological_split_with_indices(
    sequences: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    val_split: float,
    test_split: float,
):
    """
    Split chrono train/val/test (même logique que split_datasets du train),
    en conservant les indices d'origine.
    """
    assert 0 < val_split < 1 and 0 < test_split < 1 and val_split + test_split < 1
    n = len(sequences)
    if n != len(targets) or n != len(indices):
        raise ValueError("sequences, targets, indices must have the same length.")

    test_size = int(n * test_split)
    val_size = int(n * val_split)
    train_size = n - val_size - test_size
    if train_size <= 0:
        raise ValueError("Train split is empty. Check val_split and test_split.")

    train_seq = sequences[:train_size]
    val_seq = sequences[train_size : train_size + val_size]
    test_seq = sequences[train_size + val_size :]

    train_tgt = targets[:train_size]
    val_tgt = targets[train_size : train_size + val_size]
    test_tgt = targets[train_size + val_size :]

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    return (
        (train_seq, train_tgt, train_idx),
        (val_seq, val_tgt, val_idx),
        (test_seq, test_tgt, test_idx),
    )


# -----------------------------------------------------------------------------
# Modèle LSTM (sortie = LABEL_COLS)
# -----------------------------------------------------------------------------
class LSTMTLEModel(nn.Module):
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
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


# -----------------------------------------------------------------------------
# Conversion labels → éléments orbitaux physiques
# -----------------------------------------------------------------------------
def decode_angles_from_labels(
    labels_row: np.ndarray,
    label_col_index: Dict[str, int],
) -> Dict[str, float]:
    """Retourne i, raan, argp, M en radians à partir des colonnes sin/cos."""
    angles_rad = {}
    for col in ANGLE_COLS_DEG:
        base = col.replace("_deg", "")
        sin_col = f"{base}_sin"
        cos_col = f"{base}_cos"
        if sin_col not in label_col_index or cos_col not in label_col_index:
            raise ValueError(f"Missing sin/cos for angle base '{base}' in LABEL_COLS.")
        a_rad = np.arctan2(
            labels_row[label_col_index[sin_col]],
            labels_row[label_col_index[cos_col]],
        )
        angles_rad[col] = np.mod(a_rad, 2 * np.pi)
    return angles_rad


def make_satrec_from_orbital_elements(
    satnum: int,
    epoch_days: float,
    n_rev_day: float,
    ecco: float,
    inclo_rad: float,
    raan_rad: float,
    argp_rad: float,
    M_rad: float,
    bstar: float,
) -> Satrec:
    """Construit un Satrec SGP4 à partir des 7 éléments orbitaux physiques."""
    no_kozai = n_rev_day / 720.0 * np.pi  # rev/day → rad/min
    ndot = 0.0
    nddot = 0.0

    sat = Satrec()
    sat.sgp4init(
        WGS72,
        "i",
        int(satnum),
        float(epoch_days),
        bstar,
        ndot,
        nddot,
        ecco,
        argp_rad,
        inclo_rad,
        M_rad,
        no_kozai,
        raan_rad,
    )
    return sat


# -----------------------------------------------------------------------------
# Paramètres d'évaluation
# -----------------------------------------------------------------------------
@dataclass
class EvalArgs:
    csv: str
    model_path: str
    scaler_path: str
    seq_len: int
    hidden: int
    layers: int
    dropout: float
    val_split: float
    test_split: float
    seed: int
    max_samples: int
    hours: float
    step_min: float
    output_csv: str


def parse_args() -> EvalArgs:
    parser = argparse.ArgumentParser(description="Évaluer le LSTM TLE (v2) via SGP4.")
    parser.add_argument(
        "--csv",
        default="/Users/bilaldelais/Desktop/project deep learning/data/processed/iss_200000_parsed.csv",
        help="Chemin vers le CSV TLE utilisé pour l'entraînement.",
    )
    parser.add_argument(
        "--model-path",
        default="lstm_tle_model.pth",
        help="Chemin vers le modèle entraîné (state_dict).",
    )
    parser.add_argument(
        "--scaler-path",
        default="scalers.pkl",
        help="Chemin vers le fichier pickle contenant feature_scaler et label_scaler.",
    )
    parser.add_argument("--seq-len", type=int, default=20, help="Longueur des séquences L.")
    parser.add_argument("--hidden", type=int, default=128, help="Taille cachée LSTM.")
    parser.add_argument("--layers", type=int, default=2, help="Nombre de couches LSTM.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout LSTM.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Ratio validation.")
    parser.add_argument("--test-split", type=float, default=0.15, help="Ratio test.")
    parser.add_argument("--seed", type=int, default=42, help="Seed aléatoire.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Nombre max de séquences test à évaluer.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=6.0,
        help="Horizon de propagation SGP4 en heures (horizon complet).",
    )
    parser.add_argument(
        "--step-min",
        type=float,
        default=10.0,
        help="Pas de temps en minutes pour la propagation.",
    )
    parser.add_argument(
        "--output-csv",
        default="sgp4_lstm_eval_errors_v2.csv",
        help="Nom du CSV de sortie avec les erreurs (km).",
    )

    a = parser.parse_args()
    return EvalArgs(
        csv=a.csv,
        model_path=a.model_path,
        scaler_path=a.scaler_path,
        seq_len=a.seq_len,
        hidden=a.hidden,
        layers=a.layers,
        dropout=a.dropout,
        val_split=a.val_split,
        test_split=a.test_split,
        seed=a.seed,
        max_samples=a.max_samples,
        hours=a.hours,
        step_min=a.step_min,
        output_csv=a.output_csv,
    )


# Horizon court fixe pour une orbite typique de l'ISS (~90 min)
SHORT_HORIZON_MIN = 90.0


def compute_short_horizon_steps(step_min: float, full_steps: int) -> int:
    """
    Nombre de pas pour l'horizon court (90 min) en fonction du pas step_min.
    On borne par le nombre de pas de l'horizon complet.
    """
    steps_short = int(round(SHORT_HORIZON_MIN / step_min))
    if steps_short <= 0:
        steps_short = 1
    return min(steps_short, full_steps)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    global FEATURE_COLS, LABEL_COLS

    args = parse_args()
    set_seed(args.seed)

    # Device
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

    # 1) Charger données + scalers
    df = load_data(args.csv)

    with open(args.scaler_path, "rb") as f:
        scalers = pickle.load(f)

    feature_scaler: MinMaxScaler = scalers["feature_scaler"]
    label_scaler: MinMaxScaler = scalers["label_scaler"]
    FEATURE_COLS = list(scalers["feature_cols"])
    LABEL_COLS = list(scalers["label_cols"])

    print("FEATURE_COLS:", FEATURE_COLS)
    print("LABEL_COLS:", LABEL_COLS)

    # Vérification que toutes les colonnes sont présentes
    missing_feat = [c for c in FEATURE_COLS if c not in df.columns]
    missing_lab = [c for c in LABEL_COLS if c not in df.columns]
    if missing_feat:
        raise ValueError(f"Missing feature columns in DataFrame: {', '.join(missing_feat)}")
    if missing_lab:
        raise ValueError(f"Missing label columns in DataFrame: {', '.join(missing_lab)}")

    # 2) Scaling
    features = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    labels = df[LABEL_COLS].to_numpy(dtype=np.float32)

    features_scaled = feature_scaler.transform(features)
    labels_scaled = label_scaler.transform(labels)

    # 3) Construction des séquences + split chrono (identique au train)
    sequences, targets_scaled, idx_targets = build_sequences_with_indices(
        features_scaled, labels_scaled, df["sat_id"].to_numpy(), args.seq_len
    )

    (_, _, _), (_, _, _), (test_seq, test_tgt_scaled, test_idx) = chronological_split_with_indices(
        sequences, targets_scaled, idx_targets, args.val_split, args.test_split
    )

    print(f"Test sequences: {len(test_seq)}")

    # 4) Charger le modèle
    input_size = len(FEATURE_COLS)
    output_size = len(LABEL_COLS)
    model = LSTMTLEModel(
        input_size=input_size,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        output_size=output_size,
    )
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5) Prédictions sur le test
    with torch.no_grad():
        x = torch.tensor(test_seq, dtype=torch.float32, device=device)
        y_pred_scaled = model(x).cpu().numpy()

    y_true_scaled = test_tgt_scaled
    y_true_phys = label_scaler.inverse_transform(y_true_scaled)
    y_pred_phys = label_scaler.inverse_transform(y_pred_scaled)

    # -----------------------------------------------------------------------------
    # Vérification round-trip des scalers (features)
    # -----------------------------------------------------------------------------
    features_roundtrip = feature_scaler.inverse_transform(features_scaled)
    diff_feat = np.abs(features_roundtrip - features)
    feat_mae = diff_feat.mean(axis=0)
    feat_max = diff_feat.max(axis=0)
    feat_err_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "mae": feat_mae,
            "max_abs_error": feat_max,
        }
    )
    feat_err_df.to_csv("feature_errors_scaler_roundtrip_v2.csv", index=False)
    print("Round-trip scaler (features) :")
    print(feat_err_df)

    plt.figure(figsize=(10, 4))
    plt.bar(feat_err_df["feature"], feat_err_df["mae"])
    plt.xticks(rotation=90)
    plt.ylabel("MAE (unités physiques)")
    plt.title("Erreur round-trip scaler par feature (v2)")
    plt.tight_layout()
    plt.savefig("feature_errors_scaler_roundtrip_mae_v2.png")
    plt.close()

    # -----------------------------------------------------------------------------
    # Vérification round-trip des scalers (labels)
    # -----------------------------------------------------------------------------
    labels_roundtrip = label_scaler.inverse_transform(labels_scaled)
    diff_lab = np.abs(labels_roundtrip - labels)
    lab_mae = diff_lab.mean(axis=0)
    lab_max = diff_lab.max(axis=0)
    lab_err_df = pd.DataFrame(
        {
            "label": LABEL_COLS,
            "mae": lab_mae,
            "max_abs_error": lab_max,
        }
    )
    lab_err_df.to_csv("label_errors_scaler_roundtrip_v2.csv", index=False)
    print("Round-trip scaler (labels) :")
    print(lab_err_df)

    plt.figure(figsize=(10, 4))
    plt.bar(lab_err_df["label"], lab_err_df["mae"])
    plt.xticks(rotation=90)
    plt.ylabel("MAE (unités physiques)")
    plt.title("Erreur round-trip scaler par label (v2)")
    plt.tight_layout()
    plt.savefig("label_errors_scaler_roundtrip_mae_v2.png")
    plt.close()

    # -----------------------------------------------------------------------------
    # Erreurs du modèle par label (unités physiques)
    # -----------------------------------------------------------------------------
    diff_model = np.abs(y_pred_phys - y_true_phys)
    model_mae = diff_model.mean(axis=0)
    model_max = diff_model.max(axis=0)
    model_err_df = pd.DataFrame(
        {
            "label": LABEL_COLS,
            "mae": model_mae,
            "max_abs_error": model_max,
        }
    )
    model_err_df.to_csv("label_errors_model_v2.csv", index=False)
    print("Erreur de prédiction du modèle par label (v2, unités physiques) :")
    print(model_err_df)

    plt.figure(figsize=(10, 4))
    plt.bar(model_err_df["label"], model_err_df["mae"])
    plt.xticks(rotation=90)
    plt.ylabel("MAE (unités physiques)")
    plt.title("Erreur du modèle par label (v2)")
    plt.tight_layout()
    plt.savefig("label_errors_model_mae_v2.png")
    plt.close()

    # -----------------------------------------------------------------------------
    # Tests de cohérence SGP4
    # -----------------------------------------------------------------------------
    if len(test_idx) > 0:
        idx0 = int(test_idx[0])
        epoch_rel_sec0 = float(df.loc[idx0, "epoch_unix"])
        epoch_dt0 = datetime.fromtimestamp(epoch_rel_sec0, tz=timezone.utc)
        epoch_days0 = (epoch_dt0 - EPOCH0).total_seconds() / 86400.0
        satnum0 = int(df.loc[idx0, "satellite_number"])

        n0 = float(df.loc[idx0, "mean_motion_revs_per_day"])
        e0 = float(df.loc[idx0, "eccentricity"])
        i0_deg = float(df.loc[idx0, "inclination_deg"])
        raan0_deg = float(df.loc[idx0, "raan_deg"])
        argp0_deg = float(df.loc[idx0, "arg_perigee_deg"])
        M0_deg = float(df.loc[idx0, "mean_anomaly_deg"])
        bstar0 = float(df.loc[idx0, "bstar_drag"])

        i0_rad = np.deg2rad(i0_deg)
        raan0_rad = np.deg2rad(raan0_deg)
        argp0_rad = np.deg2rad(argp0_deg)
        M0_rad = np.deg2rad(M0_deg)

        # Satrec de référence
        sat_ref = make_satrec_from_orbital_elements(
            satnum0,
            epoch_days0,
            n0,
            e0,
            i0_rad,
            raan0_rad,
            argp0_rad,
            M0_rad,
            bstar0,
        )

        # 1) SGP4 vs SGP4 (même Satrec)
        errors_self = []
        n_steps_test = int(args.hours * 60.0 / args.step_min)
        for step in range(n_steps_test + 1):
            t = epoch_dt0 + timedelta(minutes=step * args.step_min)
            jd, fr = jday(
                t.year,
                t.month,
                t.day,
                t.hour,
                t.minute,
                t.second + t.microsecond / 1e6,
            )
            e1, r1, v1 = sat_ref.sgp4(jd, fr)
            e2, r2, v2 = sat_ref.sgp4(jd, fr)
            if e1 != 0 or e2 != 0:
                continue
            r1 = np.array(r1)
            r2 = np.array(r2)
            errors_self.append(np.linalg.norm(r1 - r2))
        if errors_self:
            print(
                f"[TEST] SGP4 vs SGP4 (même Satrec) - "
                f"max error = {float(np.max(errors_self)):.6e} km"
            )

        # 2) Reconstruction Satrec à partir des 7 éléments réels
        sat_rebuilt = make_satrec_from_orbital_elements(
            satnum0,
            epoch_days0,
            n0,
            e0,
            i0_rad,
            raan0_rad,
            argp0_rad,
            M0_rad,
            bstar0,
        )
        errors_rebuild = []
        for step in range(n_steps_test + 1):
            t = epoch_dt0 + timedelta(minutes=step * args.step_min)
            jd, fr = jday(
                t.year,
                t.month,
                t.day,
                t.hour,
                t.minute,
                t.second + t.microsecond / 1e6,
            )
            e1, r1, v1 = sat_ref.sgp4(jd, fr)
            e2, r2, v2 = sat_rebuilt.sgp4(jd, fr)
            if e1 != 0 or e2 != 0:
                continue
            r1 = np.array(r1)
            r2 = np.array(r2)
            errors_rebuild.append(np.linalg.norm(r1 - r2))
        if errors_rebuild:
            print(
                f"[TEST] Reconstruction Satrec (7 éléments) - "
                f"mean error = {float(np.mean(errors_rebuild)):.6e} km, "
                f"max error = {float(np.max(errors_rebuild)):.6e} km"
            )

    # -----------------------------------------------------------------------------
    # Préparation SGP4 pour les samples test
    # -----------------------------------------------------------------------------
    label_col_index = {name: i for i, name in enumerate(LABEL_COLS)}

    n_steps = int(args.hours * 60.0 / args.step_min)
    max_samples = min(args.max_samples, len(test_idx))

    print(
        f"Propager SGP4 sur {max_samples} échantillons test, "
        f"horizon {args.hours} h, pas {args.step_min} min ({n_steps+1} pas)."
    )

    results = []

    for k in range(max_samples):
        idx = int(test_idx[k])

        # Epoch réel (Unix relatif) → reconstitution datetime absolue arbitraire
        epoch_rel_sec = float(df.loc[idx, "epoch_unix"])
        epoch_dt = datetime.fromtimestamp(epoch_rel_sec, tz=timezone.utc)
        epoch_days = (epoch_dt - EPOCH0).total_seconds() / 86400.0

        # 7 éléments orbitaux "vrais" (papier) depuis le DataFrame
        n_true = float(df.loc[idx, "mean_motion_revs_per_day"])
        e_true = float(df.loc[idx, "eccentricity"])
        i_true_deg = float(df.loc[idx, "inclination_deg"])
        raan_true_deg = float(df.loc[idx, "raan_deg"])
        argp_true_deg = float(df.loc[idx, "arg_perigee_deg"])
        M_true_deg = float(df.loc[idx, "mean_anomaly_deg"])
        bstar_true = float(df.loc[idx, "bstar_drag"])

        i_true_rad = np.deg2rad(i_true_deg)
        raan_true_rad = np.deg2rad(raan_true_deg)
        argp_true_rad = np.deg2rad(argp_true_deg)
        M_true_rad = np.deg2rad(M_true_deg)

        satnum = int(df.loc[idx, "satellite_number"])

        sat_true = make_satrec_from_orbital_elements(
            satnum,
            epoch_days,
            n_true,
            e_true,
            i_true_rad,
            raan_true_rad,
            argp_true_rad,
            M_true_rad,
            bstar_true,
        )

        # 7 éléments orbitaux prédits à partir de y_pred_phys[k]
        labels_pred_row = y_pred_phys[k]
        angles_pred = decode_angles_from_labels(labels_pred_row, label_col_index)

        n_pred = float(labels_pred_row[label_col_index["mean_motion_revs_per_day"]])
        e_pred = float(labels_pred_row[label_col_index["eccentricity"]])
        bstar_pred = float(labels_pred_row[label_col_index["bstar_drag"]])

        i_pred_rad = float(angles_pred["inclination_deg"])
        raan_pred_rad = float(angles_pred["raan_deg"])
        argp_pred_rad = float(angles_pred["arg_perigee_deg"])
        M_pred_rad = float(angles_pred["mean_anomaly_deg"])

        sat_pred = make_satrec_from_orbital_elements(
            satnum,
            epoch_days,
            n_pred,
            e_pred,
            i_pred_rad,
            raan_pred_rad,
            argp_pred_rad,
            M_pred_rad,
            bstar_pred,
        )

        # Erreur de position à t0 (sans propagation dans le futur)
        jd0, fr0 = jday(
            epoch_dt.year,
            epoch_dt.month,
            epoch_dt.day,
            epoch_dt.hour,
            epoch_dt.minute,
            epoch_dt.second + epoch_dt.microsecond / 1e6,
        )
        e_true0, r_true0, v_true0 = sat_true.sgp4(jd0, fr0)
        e_pred0, r_pred0, v_pred0 = sat_pred.sgp4(jd0, fr0)
        if e_true0 != 0 or e_pred0 != 0:
            # On ne peut pas exploiter ce sample si SGP4 renvoie une erreur à t0
            continue
        r_true0 = np.array(r_true0)
        r_pred0 = np.array(r_pred0)
        error_t0_km = float(np.linalg.norm(r_true0 - r_pred0))

        # Propagation sur l'horizon complet
        errors_km = []
        for step in range(n_steps + 1):
            t = epoch_dt + timedelta(minutes=step * args.step_min)
            jd, fr = jday(
                t.year,
                t.month,
                t.day,
                t.hour,
                t.minute,
                t.second + t.microsecond / 1e6,
            )
            e1, r1, v1 = sat_true.sgp4(jd, fr)
            e2, r2, v2 = sat_pred.sgp4(jd, fr)
            if e1 != 0 or e2 != 0:
                continue

            r1 = np.array(r1)
            r2 = np.array(r2)
            diff = np.linalg.norm(r1 - r2)
            errors_km.append(diff)

        if not errors_km:
            continue

        errors_km = np.array(errors_km, dtype=float)

        # Erreurs sur l'horizon complet
        error_mean_full = float(errors_km.mean())
        error_max_full = float(errors_km.max())
        error_final_full = float(errors_km[-1])

        # Erreurs sur un horizon plus court (~1 orbite ISS, 90 min)
        steps_short = compute_short_horizon_steps(args.step_min, len(errors_km) - 1)
        errors_short = errors_km[: steps_short + 1]
        error_mean_short = float(errors_short.mean())
        error_max_short = float(errors_short.max())
        error_final_short = float(errors_short[-1])

        results.append(
            {
                "sample_index": k,
                "df_index": idx,
                "satellite_number": satnum,
                "epoch_iso": epoch_dt.isoformat(),
                "n_steps_used": len(errors_km),
                "error_t0_km": error_t0_km,
                "error_mean_km": error_mean_full,
                "error_max_km": error_max_full,
                "error_final_km": error_final_full,
                "error_mean_km_90min": error_mean_short,
                "error_max_km_90min": error_max_short,
                "error_final_km_90min": error_final_short,
            }
        )

    if not results:
        print("Aucune propagation valide (pas d'échantillons sans erreur SGP4).")
        return

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Résultats enregistrés dans {args.output_csv}")
    print(out_df.head())

    # Plots des erreurs SGP4 horizon complet
    plt.figure(figsize=(8, 4))
    plt.plot(out_df["sample_index"], out_df["error_mean_km"], label="Erreur moyenne (km)")
    plt.plot(out_df["sample_index"], out_df["error_max_km"], label="Erreur max (km)")
    plt.xlabel("Index d'échantillon test")
    plt.ylabel("Erreur (km)")
    plt.title("Erreurs SGP4 (moyenne / max) par échantillon - v2 (horizon complet)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sgp4_errors_per_sample_v2.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(out_df["error_mean_km"], bins=30)
    plt.xlabel("Erreur moyenne (km)")
    plt.ylabel("Nombre d'échantillons")
    plt.title("Distribution des erreurs SGP4 moyennes - v2 (horizon complet)")
    plt.tight_layout()
    plt.savefig("sgp4_error_mean_hist_v2.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(out_df["error_max_km"], bins=30)
    plt.xlabel("Erreur max (km)")
    plt.ylabel("Nombre d'échantillons")
    plt.title("Distribution des erreurs SGP4 max - v2 (horizon complet)")
    plt.tight_layout()
    plt.savefig("sgp4_error_max_hist_v2.png")
    plt.close()

    # Plots des erreurs SGP4 horizon court (90 min)
    plt.figure(figsize=(8, 4))
    plt.plot(
        out_df["sample_index"],
        out_df["error_mean_km_90min"],
        label="Erreur moyenne (km) - 90 min",
    )
    plt.plot(
        out_df["sample_index"],
        out_df["error_max_km_90min"],
        label="Erreur max (km) - 90 min",
    )
    plt.xlabel("Index d'échantillon test")
    plt.ylabel("Erreur (km)")
    plt.title("Erreurs SGP4 (moyenne / max) par échantillon - v2 (90 min)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sgp4_errors_per_sample_90min_v2.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(out_df["error_mean_km_90min"], bins=30)
    plt.xlabel("Erreur moyenne (km) - 90 min")
    plt.ylabel("Nombre d'échantillons")
    plt.title("Distribution des erreurs SGP4 moyennes - v2 (90 min)")
    plt.tight_layout()
    plt.savefig("sgp4_error_mean_hist_90min_v2.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(out_df["error_max_km_90min"], bins=30)
    plt.xlabel("Erreur max (km) - 90 min")
    plt.ylabel("Nombre d'échantillons")
    plt.title("Distribution des erreurs SGP4 max - v2 (90 min)")
    plt.tight_layout()
    plt.savefig("sgp4_error_max_hist_90min_v2.png")
    plt.close()


if __name__ == "__main__":
    main()