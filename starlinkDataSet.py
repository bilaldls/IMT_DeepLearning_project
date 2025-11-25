import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from skyfield.api import Loader, EarthSatellite, utc
from astroquery.jplhorizons import Horizons

# =========================
# 1) Helpers
# =========================

def tle_epoch_to_datetime(epoch_str):
    """
    epoch_str: 'YYDDD.DDDDDDDD' (ex: '25319.51552984')
    -> datetime timezone-aware (UTC)
    """
    yy = int(epoch_str[0:2])
    doy = float(epoch_str[2:])
    year = 1900 + yy if yy >= 57 else 2000 + yy
    base = datetime(year, 1, 1)
    dt = base + timedelta(days=doy - 1)
    return dt.replace(tzinfo=utc)


def parse_orbital_elements_from_line2(line2):
    """
    Extrait les éléments orbitaux depuis la ligne 2 TLE.
    """
    incl_deg = float(line2[8:16])
    raan_deg = float(line2[17:25])
    ecc_str = "0." + line2[26:33].strip()
    ecc = float(ecc_str)
    argp_deg = float(line2[34:42])
    M_deg = float(line2[43:51])
    mean_motion = float(line2[52:63])
    return incl_deg, raan_deg, ecc, argp_deg, M_deg, mean_motion


def generate_times_around_epoch(epoch_tle, window_hours=1, step_minutes=2):
    """
    Grille temporelle autour de l'epoch TLE.
    """
    times = []
    t = epoch_tle
    end = epoch_tle + timedelta(hours=window_hours)
    while t <= end:
        times.append(t)
        t += timedelta(minutes=step_minutes)
    return times


# =========================
# 2) Fichiers d'entrée
# =========================

starlink_ids_file = "/Users/leovasseur/Desktop/Projet3ADeepLearning/data/starlink_ids.csv"
starlink_tle_file = "/Users/leovasseur/Desktop/Projet3ADeepLearning/data/starlink.txt"

# =========================
# 3) Chargement du fichier IDs (name -> NORAD)
# =========================

df_ids = pd.read_csv(starlink_ids_file)

NAME_TO_NORAD = {
    str(row["satellite_name"]).strip(): int(row["norad_id"])
    for _, row in df_ids.iterrows()
}

print("Nb satellites dans starlink_ids_file :", len(NAME_TO_NORAD))


# =========================
# 4) Parsing du fichier TLE Starlink
# =========================

with open(starlink_tle_file, "r") as f:
    lines = [l.strip() for l in f if l.strip()]

records_tle = []

for i in range(0, len(lines), 3):
    name = lines[i].strip()      # ex: "STARLINK-1008"
    line1 = lines[i + 1]
    line2 = lines[i + 2]

    if name not in NAME_TO_NORAD:
        print(f"⚠️ {name} pas trouvé dans starlink_ids_file, on skip.")
        continue

    norad_id = NAME_TO_NORAD[name]

    epoch_str = line1[18:32]
    epoch_tle = tle_epoch_to_datetime(epoch_str)

    incl_deg, raan_deg, ecc, argp_deg, M_deg, mean_motion = parse_orbital_elements_from_line2(line2)

    records_tle.append({
        "name": name,
        "norad_id": norad_id,
        "line1": line1,
        "line2": line2,
        "epoch_str": epoch_str,
        "epoch_tle": epoch_tle,
        "incl_deg": incl_deg,
        "raan_deg": raan_deg,
        "ecc": ecc,
        "argp_deg": argp_deg,
        "M_deg": M_deg,
        "mean_motion_rev_per_day": mean_motion,
    })

df_tle = pd.DataFrame(records_tle)
print("TLE Starlink parsés :", len(df_tle))


# =========================
# 5) Propagation SGP4 + Horizons (NORAD = Horizons ID)
# =========================

load = Loader("data_skyfield")
ts = load.timescale()

AU_KM = 149597870.700
DAY_S = 86400.0

all_rows = []

for idx, row in df_tle.iterrows():
    name = row["name"]
    norad_id = row["norad_id"]
    line1 = row["line1"]
    line2 = row["line2"]
    epoch_tle = row["epoch_tle"]

    print(f"[{idx+1}/{len(df_tle)}] {name} (NORAD/Horizons {norad_id})")

    # Objet SGP4
    sat = EarthSatellite(line1, line2, name, ts)

    # Fenêtre temporelle (LEO -> courte)
    times_py = generate_times_around_epoch(epoch_tle, window_hours=1, step_minutes=2)
    ts_times = ts.from_datetimes(times_py)

    # --- SGP4 ---
    geo = sat.at(ts_times)
    x_sgp4, y_sgp4, z_sgp4 = geo.position.km
    vx_sgp4, vy_sgp4, vz_sgp4 = geo.velocity.km_per_s

    delta_t_min = np.array([(t - epoch_tle).total_seconds() / 60.0 for t in times_py])

    # --- Horizons : on utilise directement le NORAD comme ID ---
    try:
        jd_list = list(ts_times.tt)

        obj = Horizons(
            id=str(norad_id),   # NORAD utilisé comme Horizons ID
            id_type="id",       # on indique que c'est un identifiant numérique
            location="@399",    # géocentre
            epochs=jd_list,
        )

        vec = obj.vectors(refplane='earth')

        x_true = np.array(vec["x"]) * AU_KM
        y_true = np.array(vec["y"]) * AU_KM
        z_true = np.array(vec["z"]) * AU_KM

        vx_true = np.array(vec["vx"]) * AU_KM / DAY_S
        vy_true = np.array(vec["vy"]) * AU_KM / DAY_S
        vz_true = np.array(vec["vz"]) * AU_KM / DAY_S

        err_x = x_sgp4 - x_true
        err_y = y_sgp4 - y_true
        err_z = z_sgp4 - z_true
        err_vx = vx_sgp4 - vx_true
        err_vy = vy_sgp4 - vy_true
        err_vz = vz_sgp4 - vz_true
        err_norm = np.sqrt(err_x**2 + err_y**2 + err_z**2)

        horizons_ok = True

    except Exception as e:
        print(f"  ❌ Horizons a échoué pour {name} (NORAD {norad_id}) : {e}")
        x_true = y_true = z_true = vx_true = vy_true = vz_true = \
            err_x = err_y = err_z = err_vx = err_vy = err_vz = err_norm = \
            np.full_like(x_sgp4, np.nan)
        horizons_ok = False

    # Ajout des lignes au dataset
    for i, t_obs in enumerate(times_py):
        all_rows.append({
            "name": name,
            "norad_id": norad_id,
            "epoch_tle": epoch_tle,
            "t_obs": t_obs,
            "delta_t_min": delta_t_min[i],
            "incl_deg": row["incl_deg"],
            "raan_deg": row["raan_deg"],
            "ecc": row["ecc"],
            "argp_deg": row["argp_deg"],
            "M_deg": row["M_deg"],
            "mean_motion_rev_per_day": row["mean_motion_rev_per_day"],
            "x_sgp4": x_sgp4[i],
            "y_sgp4": y_sgp4[i],
            "z_sgp4": z_sgp4[i],
            "vx_sgp4": vx_sgp4[i],
            "vy_sgp4": vy_sgp4[i],
            "vz_sgp4": vz_sgp4[i],
            "x_true": x_true[i],
            "y_true": y_true[i],
            "z_true": z_true[i],
            "vx_true": vx_true[i],
            "vy_true": vy_true[i],
            "vz_true": vz_true[i],
            "err_x": err_x[i],
            "err_y": err_y[i],
            "err_z": err_z[i],
            "err_vx": err_vx[i],
            "err_vy": err_vy[i],
            "err_vz": err_vz[i],
            "err_norm": err_norm[i],
            "horizons_ok": horizons_ok,
        })

# =========================
# 6) DataFrame final
# =========================

df_dataset = pd.DataFrame(all_rows)
print(df_dataset.head())
print("Taille dataset Starlink (SGP4 + Horizons) :", len(df_dataset))

output_csv = "/Users/leovasseur/Desktop/Projet3ADeepLearning/data/dataset_starlink_sgp4_horizons.csv"
df_dataset.to_csv(output_csv, index=False)
print("Dataset enregistré dans :", output_csv)