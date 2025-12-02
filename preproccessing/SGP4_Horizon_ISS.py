import csv
import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil import parser as dateparser

from skyfield.api import Loader, EarthSatellite
from astroquery.jplhorizons import Horizons
from astropy.time import Time
from requests.exceptions import HTTPError

# =========================
# 1) Constantes
# =========================

CSV_TLE_FILE = "data/raw/iss_last_20_tles_spacetrack.csv"  # ton fichier epoch;name;line1;line2
HORIZONS_ID = "-125544"  # ISS
STEP_MINUTES = 5
EXTRA_HOURS_LAST = 6
OUTPUT_DATASET = "dataset_iss_sgp4_vs_horizons.csv"

MAX_RETRIES = 3          # nb de tentatives par chunk avant split ou NaN
RETRY_SLEEP_SECONDS = 3  # pause entre tentatives
MAX_CHUNK_SIZE = 25     # nombre max de dates envoyées à Horizons d'un coup

load = Loader(".")
ts = load.timescale()


# =========================
# 2) Chargement des TLE
# =========================

def load_tles_from_csv(path):
    """
    Lit un CSV epoch;name;line1;line2 et renvoie une liste triée :
    [(epoch_dt, name, l1, l2), ...]
    """
    df = pd.read_csv(path, sep=";")
    tles = []
    for _, row in df.iterrows():
        epoch_str = row["epoch"]
        epoch_dt = dateparser.parse(epoch_str)
        name = str(row["name"])
        l1 = str(row["line1"])
        l2 = str(row["line2"])
        tles.append((epoch_dt, name, l1, l2))

    tles.sort(key=lambda x: x[0])
    return tles


# =========================
# 3) Trajectoire SGP4
# =========================

def generate_sgp4_trajectory(tles, step_minutes=STEP_MINUTES,
                             extra_hours_last=EXTRA_HOURS_LAST):
    all_times = []
    all_positions = []
    all_tle_idx = []
    all_dt_since = []

    for idx, (epoch_dt, name, l1, l2) in enumerate(tles):
        sat = EarthSatellite(l1, l2, name, ts)

        if idx < len(tles) - 1:
            next_epoch_dt = tles[idx + 1][0]
        else:
            next_epoch_dt = epoch_dt + timedelta(hours=extra_hours_last)

        print(f"TLE #{idx:02d} : propagation de {epoch_dt} à {next_epoch_dt}")

        current_dt = epoch_dt
        while current_dt < next_epoch_dt:
            all_times.append(current_dt)
            all_tle_idx.append(idx)

            dt_sec = (current_dt - epoch_dt).total_seconds()
            all_dt_since.append(dt_sec)

            t_sf = ts.utc(current_dt.year, current_dt.month, current_dt.day,
                          current_dt.hour, current_dt.minute, current_dt.second)
            geo = sat.at(t_sf)
            x, y, z = geo.position.km
            all_positions.append((x, y, z))

            current_dt = current_dt + timedelta(minutes=step_minutes)

    sgp4_pos = np.array(all_positions)
    dt_since_tle = np.array(all_dt_since)
    return all_times, sgp4_pos, all_tle_idx, dt_since_tle


# =========================
# 4) Positions Horizons (robuste + chunk fixe)
# =========================

def fetch_horizons_positions(times_dt, horizons_id=HORIZONS_ID):

    print(f"Requête Horizons pour {len(times_dt)} dates")

    times_jd = Time(times_dt, scale="utc").jd
    positions = []

    def fetch_chunk(jd_chunk, depth=0):
        nonlocal positions

        if len(jd_chunk) == 0:
            return

        indent = "  " * depth
        print(f"{indent}-> chunk de {len(jd_chunk)} epochs")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                obj = Horizons(
                    id=horizons_id,
                    location="500@399",
                    epochs=jd_chunk
                )
                vec = obj.vectors()

                AU_KM = 149597870.7
                x = np.array(vec["x"]) * AU_KM
                y = np.array(vec["y"]) * AU_KM
                z = np.array(vec["z"]) * AU_KM

                for i in range(len(x)):
                    positions.append([x[i], y[i], z[i]])
                return

            except HTTPError as e:
                print(f"{indent}HTTPError tentative {attempt}/{MAX_RETRIES} : {e}")
                if attempt < MAX_RETRIES:
                    print(f"{indent}  -> pause {RETRY_SLEEP_SECONDS}s puis retry")
                    time.sleep(RETRY_SLEEP_SECONDS)
                else:
                    print(f"{indent}  -> échec après {MAX_RETRIES} tentatives")

        if len(jd_chunk) == 1:
            print(f"{indent}!! Échec définitif pour epoch {jd_chunk[0]} -> [NaN, NaN, NaN]")
            positions.append([np.nan, np.nan, np.nan])
            return

        mid = len(jd_chunk) // 2
        left = jd_chunk[:mid]
        right = jd_chunk[mid:]
        print(f"{indent}Split du chunk en {len(left)} + {len(right)} epochs...")
        fetch_chunk(left, depth + 1)
        fetch_chunk(right, depth + 1)

    n_total = len(times_jd)
    print(f"Découpage initial en chunks de taille {MAX_CHUNK_SIZE} (total {n_total} epochs)")

    for start in range(0, n_total, MAX_CHUNK_SIZE):
        end = min(start + MAX_CHUNK_SIZE, n_total)
        jd_chunk = times_jd[start:end]
        print(f"Chunk [{start}:{end}] -> {len(jd_chunk)} epochs")
        fetch_chunk(jd_chunk)

    positions = np.array(positions)
    if positions.shape[0] != n_total:
        raise RuntimeError(
            f"Incohérence : {positions.shape[0]} positions pour {n_total} dates"
        )

    return positions


# =========================
# 5) Construction du dataset
# =========================

def build_dataset():

    tles = load_tles_from_csv(CSV_TLE_FILE)
    print(f"{len(tles)} TLE chargés depuis {CSV_TLE_FILE}")

    times_dt, sgp4_pos, tle_idx, dt_since_tle = generate_sgp4_trajectory(tles)

    horizons_pos = fetch_horizons_positions(times_dt)

    if sgp4_pos.shape != horizons_pos.shape:
        raise RuntimeError("Dimensions différentes entre SGP4 et Horizons")

    diff = sgp4_pos - horizons_pos
    err_norm = np.linalg.norm(diff, axis=1)

    tle_epochs = [tles[i][0] for i in tle_idx]

    df = pd.DataFrame({
        "time_utc": [dt.isoformat() for dt in times_dt],
        "tle_index": tle_idx,
        "tle_epoch": [e.isoformat() for e in tle_epochs],
        "dt_since_tle_s": dt_since_tle,
        "x_sgp4_km": sgp4_pos[:, 0],
        "y_sgp4_km": sgp4_pos[:, 1],
        "z_sgp4_km": sgp4_pos[:, 2],
        "x_horizons_km": horizons_pos[:, 0],
        "y_horizons_km": horizons_pos[:, 1],
        "z_horizons_km": horizons_pos[:, 2],
        "dx_km": diff[:, 0],
        "dy_km": diff[:, 1],
        "dz_km": diff[:, 2],
        "error_norm_km": err_norm,
    })

    df.to_csv(OUTPUT_DATASET, sep=";", index=False)
    print(f"Dataset écrit dans {OUTPUT_DATASET}")
    print(df.head())


if __name__ == "__main__":
    build_dataset()