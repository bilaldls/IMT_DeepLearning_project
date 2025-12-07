#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Script d'évaluation : compare les TLE prédits par le LSTM avec les TLE réels
# en propageant les deux via SGP4 pour mesurer l'écart de position en km.
# -----------------------------------------------------------------------------
"""
Évaluation du modèle LSTM TLE via SGP4.

- Charge le dataset TLE (même CSV que pour l'entraînement).
- Charge le modèle LSTM + le scaler MinMax.
- Recrée les séquences, applique le même split chrono train/val/test.
- Sur la partie test uniquement, prédit le prochain TLE pour chaque séquence.
- Reconstruit deux ensembles d'éléments orbitaux (vrai vs prédit).
- Initialise deux SGP4 (Satrec) et propage sur un horizon donné.
- Calcule les erreurs de position (km) et enregistre un CSV.

Utilisation typique (depuis le dossier racine du projet) :

    python models/eval_sgp4_lstm.py \
        --csv "data/processed/iss_200000_parsed.csv" \
        --model-path "lstm_tle_model.pth" \
        --scaler-path "scaler.pkl"

"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import matplotlib.pyplot as plt

from sgp4.api import Satrec, WGS72, jday

# ---------------------------
# Colonnes et helpers communs
# ---------------------------

#
# Liste des colonnes utilisées comme features pour l'entraînement du modèle.
# Ces colonnes doivent être présentes dans le CSV et seront normalisées/dénormalisées.
#
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
    "epoch_unix",  # dérivé de "epoch"
]


# Origine temporelle utilisée par SGP4 pour convertir une date en jours fractionnaires.
EPOCH0 = datetime(1949, 12, 31, tzinfo=timezone.utc)  # origine SGP4 (jours depuis 1949-12-31)


#
# Fonction utilitaire pour fixer la seed et garantir la reproductibilité.
#
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(csv_path: str) -> pd.DataFrame:
    """Charge le CSV TLE et prépare sat_id + epoch_unix (identique à lstm_for_tle.py)."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Chargement du fichier CSV contenant l'historique des TLE.
    df = pd.read_csv(csv_path)

    required_raw_cols = ["epoch", "satellite_number"]
    missing_raw = [c for c in required_raw_cols if c not in df.columns]
    if missing_raw:
        raise ValueError(f"Missing required raw columns in CSV: {', '.join(missing_raw)}")

    # Conversion des timestamps "epoch" en secondes Unix → nécessaire pour le tri chronologique.
    # epoch_unix en secondes (gestion ISO8601/mixed)
    try:
        epoch_dt = pd.to_datetime(df["epoch"], format="ISO8601")
    except (TypeError, ValueError):
        epoch_dt = pd.to_datetime(df["epoch"], format="mixed")
    df["epoch_unix"] = epoch_dt.astype("int64") // 10**9

    df["sat_id"] = df["satellite_number"]

    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {', '.join(missing_features)}")

    df = df.sort_values(["sat_id", "epoch_unix"]).reset_index(drop=True)
    return df


#
# Construit les séquences temporelles pour le modèle LSTM tout en conservant
# l'index exact du TLE cible dans le DataFrame d'origine.
#
def build_sequences_with_indices(
    features_scaled: np.ndarray, sat_ids: Sequence, seq_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Comme build_sequences, mais renvoie aussi l'index de la cible dans le DataFrame.

    Returns
    -------
    X : (N, seq_len, D)
    y : (N, D)
    idx_target : (N,) indices dans le DataFrame original (ligne du TLE cible)
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    idxs: List[int] = []

    # Création de séquences glissantes spécifiques à chaque satellite (par sat_id).
    sat_ids = np.asarray(sat_ids)
    all_indices = np.arange(len(sat_ids))
    unique_sats = np.unique(sat_ids)

    for sat in unique_sats:
        mask = sat_ids == sat
        sat_feats = features_scaled[mask]
        sat_idx = all_indices[mask]
        if len(sat_feats) <= seq_len:
            continue
        for start in range(0, len(sat_feats) - seq_len):
            end = start + seq_len
            xs.append(sat_feats[start:end])
            ys.append(sat_feats[end])
            idxs.append(int(sat_idx[end]))

    if not xs:
        raise ValueError("No sequences could be built; check seq_len and data size.")

    return np.stack(xs), np.stack(ys), np.array(idxs, dtype=int)


#
# Réalise un split chronologique train/val/test en conservant les indices d'origine.
# Le test correspond aux TLE les plus récents dans l'historique.
#
def chronological_split_with_indices(
    sequences: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    val_split: float,
    test_split: float,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Split chrono train/val/test + indices (même logique que dans lstm_for_tle)."""
    assert 0 < val_split < 1 and 0 < test_split < 1 and val_split + test_split < 1

    n = len(sequences)
    if n != len(targets) or n != len(indices):
        raise ValueError("sequences, targets, indices must have the same length.")

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

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    return (
        (train_seq, train_tgt, train_idx),
        (val_seq, val_tgt, val_idx),
        (test_seq, test_tgt, test_idx),
    )


#
# Modèle LSTM utilisé pour prédire les paramètres orbitaux du prochain TLE.
#
class LSTMTLEModel(nn.Module):
    """Même architecture que dans lstm_for_tle.py."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
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
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


def make_satrec_from_features(
    feats: np.ndarray,
    col_index: dict,
    satnum: int,
    epoch_days: float,
) -> Satrec:
    """
    Construit un Satrec SGP4 à partir d'un vecteur de features physiques (non normalisées).

    Conversion inspirée de sgp4/omm.py :
    - MEAN_MOTION (rev/day) → no_kozai (rad/min) = MM / 720 * pi
    - INCLINATION / RAAN / ARG_PERIGEE / MEAN_ANOMALY (deg) → radians
    - bstar directement en 1/earth radii
    - ndot, nddot mis à 0 (SGP4 les ignore de toute façon).
    """
    # Conversion des paramètres orbitaux prédits/non normalisés
    # vers les unités requises par SGP4 (radians, rad/min, etc.).
    ecco = float(feats[col_index["eccentricity"]])
    inclo = np.deg2rad(float(feats[col_index["inclination_deg"]]))
    nodeo = np.deg2rad(float(feats[col_index["raan_deg"]]))
    argpo = np.deg2rad(float(feats[col_index["arg_perigee_deg"]]))
    mo = np.deg2rad(float(feats[col_index["mean_anomaly_deg"]]))
    mean_motion_rev_day = float(feats[col_index["mean_motion_revs_per_day"]])
    no_kozai = mean_motion_rev_day / 720.0 * np.pi  # rev/day → rad/min
    bstar = float(feats[col_index["bstar_drag"]])
    ndot = 0.0
    nddot = 0.0

    # Initialisation du propagateur SGP4 avec les éléments orbitaux.
    sat = Satrec()
    sat.sgp4init(
        WGS72,
        "i",          # mode "improved"
        int(satnum),  # satnum
        float(epoch_days),
        bstar,
        ndot,
        nddot,
        ecco,
        argpo,
        inclo,
        mo,
        no_kozai,
        nodeo,
    )
    return sat


#
# Paramètres configurables pour l'évaluation SGP4 (séquences, horizon, etc.).
#
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
    # Définition des arguments de la ligne de commande.
    parser = argparse.ArgumentParser(description="Évaluer le LSTM TLE via SGP4.")
    parser.add_argument(
        "--csv",
        default="/Users/bilaldelais/Desktop/project deep learning/data/processed/iss_200000_parsed.csv",
        help="Chemin vers le CSV TLE utilisé pour l'entraînement (ex: iss_200000_parsed.csv).",
    )
    # Définition des arguments de la ligne de commande.
    parser.add_argument(
        "--model-path",
        default="lstm_tle_model.pth",
        help="/Users/bilaldelais/Desktop/project deep learning/models/config/lstm_tle_model.pth",
    )
    # Définition des arguments de la ligne de commande.
    parser.add_argument(
        "--scaler-path",
        default="scaler.pkl",
        help="/Users/bilaldelais/Desktop/project deep learning/models/config/scaler.pkl",
    )
    # Définition des arguments de la ligne de commande.
    parser.add_argument("--seq-len", type=int, default=20, help="Longueur des séquences L.")
    parser.add_argument("--hidden", type=int, default=128, help="Taille cachée LSTM.")
    parser.add_argument("--layers", type=int, default=2, help="Nombre de couches LSTM.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout LSTM.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Ratio validation.")
    parser.add_argument("--test-split", type=float, default=0.15, help="Ratio test.")
    parser.add_argument("--seed", type=int, default=42, help="Seed aléatoire.")
    # Définition des arguments de la ligne de commande.
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Nombre max de séquences test à évaluer (pour éviter un temps trop long).",
    )
    # Définition des arguments de la ligne de commande.
    parser.add_argument(
        "--hours",
        type=float,
        default=6.0,
        help="Horizon de propagation SGP4 en heures à partir de l'epoch du TLE.",
    )
    # Définition des arguments de la ligne de commande.
    parser.add_argument(
        "--step-min",
        type=float,
        default=10.0,
        help="Pas de temps en minutes pour la propagation SGP4.",
    )
    # Définition des arguments de la ligne de commande.
    parser.add_argument(
        "--output-csv",
        default="sgp4_lstm_eval_errors.csv",
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


def main() -> None:
    # ----------------------
    # Début du pipeline d'évaluation :
    #  - Chargement des données et du scaler
    #  - Reconstruction des séquences test
    #  - Chargement du modèle LSTM entraîné
    #  - Prédiction des futurs TLE
    #  - Propagation SGP4 pour mesurer l'erreur en km
    # ----------------------
    args = parse_args()
    set_seed(args.seed)

    # Device (MPS / CUDA / CPU)
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

    # 1) Charger les données et scaler
    df = load_data(args.csv)
    with open(args.scaler_path, "rb") as f:
        scaler: MinMaxScaler = pickle.load(f)

    # Les données physiques doivent être retransformées dans l'espace réduit [0,1]
    # car le modèle a été entraîné dans cet espace normalisé.
    features_scaled = scaler.transform(df[FEATURE_COLS].to_numpy(dtype=np.float32))
    sequences, targets, idx_targets = build_sequences_with_indices(
        features_scaled, df["sat_id"].to_numpy(), args.seq_len
    )

    # Split chronologique identique à celui utilisé pendant l'entraînement.
    # 2) Split chrono train/val/test (on ne garde que la partie test)
    (_, _, _), (_, _, _), (test_seq, test_tgt_scaled, test_idx) = chronological_split_with_indices(
        sequences, targets, idx_targets, args.val_split, args.test_split
    )

    print(f"Test sequences: {len(test_seq)}")

    # 3) Charger le modèle
    input_size = test_seq.shape[-1]
    model = LSTMTLEModel(
        input_size=input_size,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    )
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Prédiction du prochain TLE pour chaque séquence test.
    # 4) Prédictions sur tout le test (en un bloc pour simplifier)
    with torch.no_grad():
        x = torch.tensor(test_seq, dtype=torch.float32, device=device)
        y_pred_scaled = model(x).cpu().numpy()

    # Retour dans l'espace physique : les valeurs redeviennent exploitées par SGP4.
    y_true_scaled = test_tgt_scaled
    y_true = scaler.inverse_transform(y_true_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # -------------------------------
    # Vérification 1 : round-trip scaler
    # -------------------------------
    # On compare y_true (features reconstruits après normalisation/inverse_transform)
    # aux valeurs physiques originales du DataFrame pour les mêmes lignes.
    df_test_features = df.loc[test_idx, FEATURE_COLS].to_numpy(dtype=np.float32)
    diff_scaler = np.abs(y_true - df_test_features)
    scaler_mae = diff_scaler.mean(axis=0)
    scaler_max = diff_scaler.max(axis=0)

    scaler_err_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "mae": scaler_mae,
            "max_abs_error": scaler_max,
        }
    )
    scaler_err_df.to_csv("feature_errors_scaler_roundtrip.csv", index=False)
    print("Erreur round-trip scaler (y_true vs df[FEATURE_COLS]) :")
    print(scaler_err_df)

    # Plot des erreurs round-trip du scaler (MAE par feature)
    plt.figure(figsize=(10, 4))
    plt.bar(scaler_err_df["feature"], scaler_err_df["mae"])
    plt.xticks(rotation=90)
    plt.ylabel("MAE (unités physiques)")
    plt.title("Erreur round-trip scaler par feature")
    plt.tight_layout()
    plt.savefig("feature_errors_scaler_roundtrip_mae.png")
    plt.close()

    # -------------------------------
    # Vérification 2 : erreurs par feature du modèle (en unités physiques)
    # -------------------------------
    diff_model = np.abs(y_pred - y_true)
    model_mae = diff_model.mean(axis=0)
    model_max = diff_model.max(axis=0)

    model_err_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "mae": model_mae,
            "max_abs_error": model_max,
        }
    )
    model_err_df.to_csv("feature_errors_model.csv", index=False)
    print("Erreur de prédiction du modèle par feature (en unités physiques) :")
    print(model_err_df)

    # Plot des erreurs du modèle par feature (MAE)
    plt.figure(figsize=(10, 4))
    plt.bar(model_err_df["feature"], model_err_df["mae"])
    plt.xticks(rotation=90)
    plt.ylabel("MAE (unités physiques)")
    plt.title("Erreur du modèle par feature")
    plt.tight_layout()
    plt.savefig("feature_errors_model_mae.png")
    plt.close()

    # Index des colonnes pour accéder rapidement aux features
    col_index = {name: i for i, name in enumerate(FEATURE_COLS)}

    # -------------------------------
    # Vérification 3 : cohérence SGP4
    # -------------------------------
    # On vérifie que la reconstruction SGP4 à partir de :
    #   - df[FEATURE_COLS] (valeurs brutes)
    #   - y_true (valeurs après scaler + inverse_transform)
    # donne des positions quasi identiques au temps epoch.
    sgp4_self_results = []
    check_samples = min(5, len(test_idx))

    for k in range(check_samples):
        idx = int(test_idx[k])

        # Features brutes issues du DataFrame
        feats_df = df.loc[idx, FEATURE_COLS].to_numpy(dtype=float)
        # Features reconstruites après passage par le scaler
        feats_ref = y_true[k]

        epoch_sec = float(df.loc[idx, "epoch_unix"])
        epoch_dt = datetime.fromtimestamp(epoch_sec, tz=timezone.utc)
        epoch_days = (epoch_dt - EPOCH0).total_seconds() / 86400.0

        satnum = int(round(feats_df[col_index["satellite_number"]]))

        sat_a = make_satrec_from_features(feats_df, col_index, satnum, epoch_days)
        sat_b = make_satrec_from_features(feats_ref, col_index, satnum, epoch_days)

        jd, fr = jday(
            epoch_dt.year,
            epoch_dt.month,
            epoch_dt.day,
            epoch_dt.hour,
            epoch_dt.minute,
            epoch_dt.second + epoch_dt.microsecond / 1e6,
        )
        e1, r1, v1 = sat_a.sgp4(jd, fr)
        e2, r2, v2 = sat_b.sgp4(jd, fr)

        if e1 == 0 and e2 == 0:
            r1 = np.array(r1)
            r2 = np.array(r2)
            err0 = np.linalg.norm(r1 - r2)
            sgp4_self_results.append(err0)

    if sgp4_self_results:
        print(
            "Erreur SGP4 (reconstruction scaler vs données brutes) au temps epoch : "
            f"max={max(sgp4_self_results):.6f} km, "
            f"mean={np.mean(sgp4_self_results):.6f} km"
        )

    # Boucle principale d'évaluation :
    # pour chaque TLE test → reconstruire deux Satrec (vrai et prédit) → propager.
    # 5) Boucle SGP4 : pour chaque échantillon test, comparer position vraie vs prédite
    n_steps = int(args.hours * 60.0 / args.step_min)
    max_samples = min(args.max_samples, len(test_idx))

    print(
        f"Propager SGP4 sur {max_samples} échantillons test, "
        f"horizon {args.hours} h, pas {args.step_min} min ({n_steps+1} pas)."
    )

    results = []
    for k in range(max_samples):
        idx = int(test_idx[k])

        # Epoch réel (en secondes depuis 1970) puis conversion
        epoch_sec = float(df.loc[idx, "epoch_unix"])
        epoch_dt = datetime.fromtimestamp(epoch_sec, tz=timezone.utc)
        epoch_days = (epoch_dt - EPOCH0).total_seconds() / 86400.0

        feats_true = y_true[k]
        feats_pred = y_pred[k]

        satnum_true = int(round(feats_true[col_index["satellite_number"]]))
        satnum_pred = int(round(feats_pred[col_index["satellite_number"]]))

        # On force le même satnum pour les deux (dans ton cas ISS => 25544)
        satnum = satnum_true

        sat_true = make_satrec_from_features(feats_true, col_index, satnum, epoch_days)
        sat_pred = make_satrec_from_features(feats_pred, col_index, satnum, epoch_days)

        errors_km = []
        times = []

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

            # On ignore les pas où SGP4 renvoie une erreur
            if e1 != 0 or e2 != 0:
                continue

            # r1 et r2 sont les vecteurs position en km ; leur différence donne l'erreur spatiale.
            r1 = np.array(r1)  # km
            r2 = np.array(r2)  # km
            diff = np.linalg.norm(r1 - r2)
            errors_km.append(diff)
            times.append(t)

        if not errors_km:
            continue

        errors_km = np.array(errors_km, dtype=float)
        results.append(
            {
                "sample_index": k,
                "df_index": idx,
                "satellite_number": satnum,
                "epoch_iso": epoch_dt.isoformat(),
                "n_steps_used": len(errors_km),
                "error_mean_km": float(errors_km.mean()),
                "error_max_km": float(errors_km.max()),
                "error_final_km": float(errors_km[-1]),
            }
        )

    if not results:
        print("Aucune propagation valide (pas d'échantillons sans erreur SGP4).")
        return

    # Sauvegarde des métriques d'erreur (moyenne, max, finale) dans un CSV.
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Résultats enregistrés dans {args.output_csv}")
    print(out_df.head())

    # -------------------------------
    # Plots des erreurs SGP4 (position en km)
    # -------------------------------
    # Courbe des erreurs moyennes et max en fonction de l'index d'échantillon
    plt.figure(figsize=(8, 4))
    plt.plot(out_df["sample_index"], out_df["error_mean_km"], label="Erreur moyenne (km)")
    plt.plot(out_df["sample_index"], out_df["error_max_km"], label="Erreur max (km)")
    plt.xlabel("Index d'échantillon test")
    plt.ylabel("Erreur (km)")
    plt.title("Erreurs SGP4 (moyenne / max) par échantillon")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sgp4_errors_per_sample.png")
    plt.close()

    # Histogramme des erreurs moyennes
    plt.figure(figsize=(8, 4))
    plt.hist(out_df["error_mean_km"], bins=30)
    plt.xlabel("Erreur moyenne (km)")
    plt.ylabel("Nombre d'échantillons")
    plt.title("Distribution des erreurs SGP4 moyennes")
    plt.tight_layout()
    plt.savefig("sgp4_error_mean_hist.png")
    plt.close()

    # Histogramme des erreurs max
    plt.figure(figsize=(8, 4))
    plt.hist(out_df["error_max_km"], bins=30)
    plt.xlabel("Erreur max (km)")
    plt.ylabel("Nombre d'échantillons")
    plt.title("Distribution des erreurs SGP4 max")
    plt.tight_layout()
    plt.savefig("sgp4_error_max_hist.png")
    plt.close()


if __name__ == "__main__":
    main()