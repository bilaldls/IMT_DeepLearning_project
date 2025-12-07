#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")  # backend non interactif pour usage CLI
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# =============================
# 🔧 PARAMÈTRES À PERSONNALISER
# =============================

FILES = [
    "Erreur_MLP_LR0.0001_HL512-256-128.csv",
    "Erreur_LSTM_LR0.0001_HL128_L3.csv",
    "Erreur_GRU_LR0.0001_HL128_L1.csv"

]

TIME_COLUMN = None  # ex: "dt_since_tle_s"

# Seuil pour masquer les pics
MAX_ERR_TO_PLOT_KM = 50.0
LAST_N_POINTS = 93  # nombre de valeurs conservées (fin de série)


# =============================
# 🔁 FONCTIONS UTILITAIRES
# =============================

def load_df(path: Path):
    df = pd.read_csv(path)

    required = [
        "x_sgp4_km", "y_sgp4_km", "z_sgp4_km",
        "x_horizons_km", "y_horizons_km", "z_horizons_km",
        "err_x_pred", "err_y_pred", "err_z_pred",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {', '.join(missing)}")

    if TIME_COLUMN is not None and TIME_COLUMN in df.columns:
        t = df[TIME_COLUMN]
    else:
        t = df.index

    df["err_x_true"] = df["x_horizons_km"] - df["x_sgp4_km"]
    df["err_y_true"] = df["y_horizons_km"] - df["y_sgp4_km"]
    df["err_z_true"] = df["z_horizons_km"] - df["z_sgp4_km"]

    df["err_norm_true"] = np.sqrt(
        df["err_x_true"]**2 + df["err_y_true"]**2 + df["err_z_true"]**2
    )
    df["err_norm_pred"] = np.sqrt(
        df["err_x_pred"]**2 + df["err_y_pred"]**2 + df["err_z_pred"]**2
    )

    return df, t


# =============================
# 📈 GRAPHIQUE UNIQUE
# =============================

def plot_error_norms(df: pd.DataFrame,
                     t,
                     ax,
                     label_prefix: str,
                     max_err: Optional[float] = None) -> None:

    df_plot = df.copy()

    if max_err is not None:
        df_plot.loc[df_plot["err_norm_true"] > max_err, "err_norm_true"] = np.nan
        df_plot.loc[df_plot["err_norm_pred"] > max_err, "err_norm_pred"] = np.nan

    ax.plot(t, df_plot["err_norm_true"], label=f"{label_prefix} ||err_true||")
    ax.plot(t, df_plot["err_norm_pred"], label=f"{label_prefix} ||err_pred||", linestyle="--")


def save_legend(handles, labels, outpath: Path):
    fig_legend = plt.figure(figsize=(10, 2))
    fig_legend.legend(handles, labels, loc="center", ncol=2)
    fig_legend.tight_layout()
    fig_legend.savefig(outpath, dpi=300)
    plt.close(fig_legend)


# =============================
# 🚀 MAIN
# =============================

def main():
    if not FILES:
        raise SystemExit("❌ La liste FILES est vide.")

    fig, ax = plt.subplots(figsize=(12, 6))

    for fname in FILES:
        path = Path(fname)
        if not path.exists():
            raise SystemExit(f"❌ Fichier introuvable : {path}")

        print(f"📄 Traitement de : {path}")

        df, t = load_df(path)
        if LAST_N_POINTS:
            df = df.tail(LAST_N_POINTS)
            t = t[-LAST_N_POINTS:]
        stem = path.stem

        plot_error_norms(df, t, ax, stem, max_err=MAX_ERR_TO_PLOT_KM)

    ax.set_xlabel(TIME_COLUMN if TIME_COLUMN else "index")
    ax.set_ylabel("Norme de l'erreur (km)")
    title_suffix = f" (derniers {LAST_N_POINTS} points)" if LAST_N_POINTS else ""
    ax.set_title(f"Norme des erreurs{title_suffix}")
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    out_err_norm = Path("errors_norm_all_filtered.png")
    fig.savefig(out_err_norm, dpi=300)
    plt.close(fig)

    legend_out = Path("errors_norm_all_filtered_legend.png")
    save_legend(handles, labels, legend_out)

    print(f"\n✅ Graphique généré : {out_err_norm}")
    print(f"✅ Légende séparée : {legend_out}")


if __name__ == "__main__":
    main()
