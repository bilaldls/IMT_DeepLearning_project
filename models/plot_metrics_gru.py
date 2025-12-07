#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")  # backend non interactif
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# =============================
# 🔧 PARAMÈTRES À PERSONNALISER
# =============================

FILES = [
    "Metrics_GRU_LR0.001_HL64_L1.csv",
    "Metrics_GRU_LR0.0001_HL64_L3.csv",
    "Metrics_LSTM_LR0.0001_HL128_L3.csv",
    "Metrics_LSTM_LR5e-05_HL128_L3.csv",
]

X_COLUMN = "Epoch"        # colonne en abscisse
Y_COLUMN = "Val_MSE"    # colonne en ordonnée
OUTPUT = "courbes_bestGRUvsbestLSTM_epoch_Val_MSE.png"  # fichier de sortie


# =============================

def load_xy(path: Path):
    df = pd.read_csv(path)
    if X_COLUMN not in df.columns or Y_COLUMN not in df.columns:
        missing = [col for col in (X_COLUMN, Y_COLUMN) if col not in df.columns]
        raise ValueError(f"{path.name} missing columns: {', '.join(missing)}")
    return df[X_COLUMN], df[Y_COLUMN]


def plot_metrics(files):
    plt.figure(figsize=(12, 7))

    for path in files:
        x, y = load_xy(path)
        plt.plot(x, y, label=path.stem)

    plt.xlabel(X_COLUMN)
    plt.ylabel(Y_COLUMN)
    plt.title(f"{Y_COLUMN} en fonction de {X_COLUMN}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Ajouter un encadré avec la liste des fichiers
    file_lines = [f"- {str(path)}" for path in files]
    info_text = "Fichiers utilisés :\n" + "\n".join(file_lines)

    plt.gcf().text(
        0.01, 0.01,
        info_text,
        fontsize=6,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )


def main():
    if not FILES:
        raise SystemExit("❌ La liste FILES est vide. Ajoute tes CSV dans FILES.")

    # Convertir en objets Path et vérifier leur existence
    paths = []
    for f in FILES:
        p = Path(f)
        if not p.exists():
            raise SystemExit(f"❌ Fichier introuvable : {p}")
        paths.append(p)

    print("📄 Fichiers sélectionnés :")
    for p in paths:
        print("  •", p)

    plot_metrics(paths)

    plt.savefig(OUTPUT, dpi=300)
    print(f"\n📊 Graphique sauvegardé dans : {OUTPUT}")

    plt.show()


if __name__ == "__main__":
    main()