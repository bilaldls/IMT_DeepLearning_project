import numpy as np
import pandas as pd
from sgp4.api import Satrec
from datetime import datetime, timedelta
from astroquery.jplhorizons import Horizons, Conf
from astropy.time import Time
from tqdm import tqdm  # barre de progression

# Timeout pour Horizons
Conf.timeout = 120

TLE_FILE = "/Users/leovasseur/Desktop/Projet3ADeepLearning/zaryaISS.txt"
OUTPUT_CSV = "ISS.csv"

# ID Horizons de TERRA (EOS-1) : -125994
HORIZONS_ID = "-125544"


# ------------------------------------------------------------
# 1) Lecture et parsing des TLE (nom optionnel + 2 lignes)
# ------------------------------------------------------------
def read_tle_file(path):
    records = []
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") or lines[i].startswith("2 "):
            line1 = lines[i]
            line2 = lines[i + 1]
            name = ""
            i += 2
        else:
            name = lines[i]
            line1 = lines[i + 1]
            line2 = lines[i + 2]
            i += 3

        records.append({"name": name, "line1": line1, "line2": line2})

    return records


# ------------------------------------------------------------
# 2) Conversion epoch TLE -> datetime + JD
# ------------------------------------------------------------
def tle_epoch_to_datetime(line1):
    epoch_str = line1[18:32].strip()
    yy = int(epoch_str[0:2])
    day_of_year = float(epoch_str[2:])

    year = 2000 + yy if yy < 57 else 1900 + yy

    day_int = int(day_of_year)
    frac_day = day_of_year - day_int

    date0 = datetime(year, 1, 1) + timedelta(days=day_int - 1)
    return date0 + timedelta(days=frac_day)


def datetime_to_jd(dt):
    return Time(dt, scale="utc").tdb.jd


# ------------------------------------------------------------
# 3) Extraction des éléments orbitaux numériques (ligne 2)
# ------------------------------------------------------------
def extract_orbital_elements_from_line2(line2):
    incl_deg = float(line2[8:16])
    raan_deg = float(line2[17:25])
    ecc = float("0." + line2[26:33].strip())
    argp_deg = float(line2[34:42])
    M_deg = float(line2[43:51])
    n_rev_per_day = float(line2[52:63])
    return incl_deg, raan_deg, ecc, argp_deg, M_deg, n_rev_per_day


# ------------------------------------------------------------
# 4) Construire DataFrame avec éléments TLE + SGP4
# ------------------------------------------------------------
def build_records_with_sgp4(tle_records):
    data = []

    print("Propagation SGP4 + extraction des éléments orbitaux…")
    for rec in tqdm(tle_records, desc="SGP4", unit="tle"):
        line1 = rec["line1"]
        line2 = rec["line2"]

        # Epoch
        epoch_dt = tle_epoch_to_datetime(line1)
        jd = datetime_to_jd(epoch_dt)

        # SGP4 à l'époque du TLE
        sat = Satrec.twoline2rv(line1, line2)
        jd_int = int(jd)
        jd_fr = jd - jd_int
        e, r, v = sat.sgp4(jd_int, jd_fr)
        if e != 0:
            continue  # on skippe les TLE foireux

        x, y, z = r
        vx, vy, vz = v

        incl_deg, raan_deg, ecc, argp_deg, M_deg, n_rev = extract_orbital_elements_from_line2(line2)

        row = {
            "epoch_utc": epoch_dt,
            "epoch_jd_tdb": jd,
            "incl_deg": incl_deg,
            "raan_deg": raan_deg,
            "ecc": ecc,
            "argp_deg": argp_deg,
            "M_deg": M_deg,
            "mean_motion_rev_per_day": n_rev,
            "sgp4_x_km": x,
            "sgp4_y_km": y,
            "sgp4_z_km": z,
            "sgp4_vx_km_s": vx,
            "sgp4_vy_km_s": vy,
            "sgp4_vz_km_s": vz,
        }

        data.append(row)

    return pd.DataFrame(data)


# ------------------------------------------------------------
# 5) Appel à Horizons (chunks + retries, géocentrique)
# ------------------------------------------------------------
def query_horizons_vectors(df, chunk_size=50, max_retries=3):
    jd_array = df["epoch_jd_tdb"].to_numpy()
    n = len(jd_array)

    true_x = np.full(n, np.nan)
    true_y = np.full(n, np.nan)
    true_z = np.full(n, np.nan)
    true_vx = np.full(n, np.nan)
    true_vy = np.full(n, np.nan)
    true_vz = np.full(n, np.nan)

    AU_KM = 149597870.700
    sat_id = HORIZONS_ID.strip()

    print("Requête Horizons (géocentrique) sur tous les epochs…")
    for start in tqdm(range(0, n, chunk_size), desc="Horizons", unit="chunk"):
        end = min(start + chunk_size, n)
        jd_chunk = jd_array[start:end].tolist()

        success = False
        attempt = 0

        while not success and attempt < max_retries:
            attempt += 1
            try:
                obj = Horizons(
                    id=sat_id,
                    location="500@399",   # géocentrique
                    epochs=jd_chunk,
                    id_type="id",         # on donne l'ID direct -125994
                )

                vec = obj.vectors()

                x_au = np.array(vec["x"])
                y_au = np.array(vec["y"])
                z_au = np.array(vec["z"])
                vx_au_d = np.array(vec["vx"])
                vy_au_d = np.array(vec["vy"])
                vz_au_d = np.array(vec["vz"])

                true_x[start:end] = x_au * AU_KM
                true_y[start:end] = y_au * AU_KM
                true_z[start:end] = z_au * AU_KM
                true_vx[start:end] = vx_au_d * AU_KM / 86400.0
                true_vy[start:end] = vy_au_d * AU_KM / 86400.0
                true_vz[start:end] = vz_au_d * AU_KM / 86400.0

                success = True

            except Exception as e:
                print(f"\n⚠️ Erreur Horizons chunk {start}:{end}, tentative {attempt}/{max_retries}")
                print(e)
                if attempt == max_retries:
                    print("⛔ Chunk échoué, on laisse NaN pour ces lignes.")

    df["true_x_km"] = true_x
    df["true_y_km"] = true_y
    df["true_z_km"] = true_z
    df["true_vx_km_s"] = true_vx
    df["true_vy_km_s"] = true_vy
    df["true_vz_km_s"] = true_vz

    return df


# ------------------------------------------------------------
# 6) Main – fichier complet
# ------------------------------------------------------------
def main():
    print("Lecture TLE…")
    tle_records = read_tle_file(TLE_FILE)
    print(f"{len(tle_records)} TLE trouvés dans le fichier")

    print("SGP4 propagation + extraction des éléments…")
    df = build_records_with_sgp4(tle_records)
    print(f"{len(df)} lignes valides après SGP4")

    print("Appel Horizons…")
    df = query_horizons_vectors(df, chunk_size=50, max_retries=3)

    print("Sauvegarde CSV final…")
    df.to_csv(OUTPUT_CSV, index=False)
    print("✅ Dataset généré :", OUTPUT_CSV)


if __name__ == "__main__":
    main()