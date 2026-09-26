"""
Corrige, sans nouvelle requête Horizons, les datasets générés avant le
correctif de sgp4_TLE_horizons.py (troncature des microsecondes du temps de
propagation SGP4).

Les positions Horizons ont été demandées à l'instant exact : elles restent
valides. Seules les colonnes SGP4 (et donc dx/dy/dz/error_norm) sont fausses.
Pour chaque TLE, le satellite SGP4 est reconstruit à partir des paramètres
orbitaux stockés dans le dataset, puis :
  1) on vérifie qu'il reproduit les anciennes positions à l'instant tronqué
     (sinon on s'arrête : la reconstruction ne serait pas fidèle) ;
  2) on recalcule les positions à l'instant exact.

Exemple :
    python preproccessing/fix_time_truncation.py data/datasetISS_200TLE.csv
"""
import argparse
import math

import numpy as np
import pandas as pd
from sgp4.api import Satrec, WGS72
from skyfield.api import EarthSatellite, load

ts = load.timescale(builtin=True)

# Écart max toléré entre ancienne position et reconstruction à l'instant tronqué
MAX_REPRODUCTION_ERROR_KM = 1e-3


def satellite_from_params(row) -> EarthSatellite:
    """Reconstruit un satellite SGP4 à partir des colonnes TLE du dataset."""
    epoch_jd = pd.Timestamp(row.tle_epoch).to_julian_date()
    satrec = Satrec()
    satrec.sgp4init(
        WGS72,
        "i",
        0,
        epoch_jd - 2433281.5,  # jours depuis 1949-12-31 00:00 UT
        row.bstar,
        row.mean_motion_derivative * 2 * math.pi / 1440 ** 2,  # rev/jour^2 -> rad/min^2
        0.0,
        row.eccentricity,
        math.radians(row.arg_perigee_deg),
        math.radians(row.inclination_deg),
        math.radians(row.mean_anomaly_deg),
        row.mean_motion * 2 * math.pi / 1440,  # rev/jour -> rad/min
        math.radians(row.raan_deg),
    )
    return EarthSatellite.from_satrec(satrec, ts)


def fix_dataset(path: str, output: str) -> None:
    df = pd.read_csv(path, sep=";", float_precision="round_trip")
    times = pd.to_datetime(df["time_utc"], format="ISO8601")

    old_pos = df[["x_sgp4_km", "y_sgp4_km", "z_sgp4_km"]].to_numpy()
    new_pos = np.empty_like(old_pos)
    reproduction_err = np.empty(len(df))

    for _, group in df.groupby("tle_index", sort=False):
        sat = satellite_from_params(group.iloc[0])
        idx = group.index.to_numpy()
        t = times[idx]

        t_trunc = ts.utc(t.dt.year.to_numpy(), t.dt.month.to_numpy(), t.dt.day.to_numpy(),
                         t.dt.hour.to_numpy(), t.dt.minute.to_numpy(), t.dt.second.to_numpy())
        t_exact = ts.from_datetimes(t.dt.to_pydatetime())

        reproduced = sat.at(t_trunc).position.km.T
        reproduction_err[idx] = np.linalg.norm(reproduced - old_pos[idx], axis=1)
        new_pos[idx] = sat.at(t_exact).position.km.T

    worst = reproduction_err.max()
    print(f"{path} : reproduction des anciennes positions, écart max = {worst * 1000:.2f} m")
    if worst > MAX_REPRODUCTION_ERROR_KM:
        raise SystemExit("Reconstruction SGP4 non fidèle : dataset laissé intact.")

    horizons = df[["x_horizons_km", "y_horizons_km", "z_horizons_km"]].to_numpy()
    diff = new_pos - horizons
    old_err = df["error_norm_km"].to_numpy()

    df[["x_sgp4_km", "y_sgp4_km", "z_sgp4_km"]] = new_pos
    df[["dx_km", "dy_km", "dz_km"]] = diff
    df["error_norm_km"] = np.linalg.norm(diff, axis=1)

    print(f"  erreur médiane : {np.nanmedian(old_err):.3f} km -> {df['error_norm_km'].median():.3f} km")
    df.to_csv(output, sep=";", index=False)
    print(f"  écrit dans {output}")


def main():
    parser = argparse.ArgumentParser(description="Corrige la troncature temporelle SGP4 d'un dataset existant")
    parser.add_argument("datasets", nargs="+", help="CSV produits par sgp4_TLE_horizons.py (corrigés sur place)")
    args = parser.parse_args()
    for path in args.datasets:
        fix_dataset(path, path)


if __name__ == "__main__":
    main()
