"""
Télécharge les N derniers TLE d'un satellite depuis Space-Track et les écrit
au format CSV epoch;name;line1;line2.

Les identifiants sont lus dans les variables d'environnement
SPACETRACK_USER et SPACETRACK_PASS (voir .env.example) : ne jamais les
écrire dans le code.

Exemple :
    export SPACETRACK_USER=... SPACETRACK_PASS=...
    python preproccessing/fetch_tles.py --norad 25544 --name ISS --count 2000 \
        --output data/raw/iss_last_2000_tles_spacetrack.csv
"""
import argparse
import csv
import math
import os
from datetime import datetime, timedelta, timezone

import requests

BASE_URL = "https://www.space-track.org"
LOGIN_URL = BASE_URL + "/ajaxauth/login"


def get_credentials():
    user = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASS")
    if not user or not password:
        raise SystemExit(
            "Identifiants Space-Track manquants : définir SPACETRACK_USER et "
            "SPACETRACK_PASS dans l'environnement (voir .env.example)."
        )
    return user, password


def parse_tle_epoch(epoch_str: str) -> datetime:
    """
    Convertit l'époque TLE (YYDDD.DDDDD) en datetime UTC.
    """
    epoch_str = epoch_str.strip()
    yy = int(epoch_str[:2])
    doy = float(epoch_str[2:])
    year = 1900 + yy if yy >= 57 else 2000 + yy

    day_int = int(math.floor(doy))
    frac = doy - day_int

    dt0 = datetime(year, 1, 1, tzinfo=timezone.utc)
    return dt0 + timedelta(days=day_int - 1, seconds=frac * 86400)


def tle_query_url(norad_id: int, count: int) -> str:
    return (
        BASE_URL
        + "/basicspacedata/query/"
        "class/tle/"
        f"NORAD_CAT_ID/{norad_id}/"
        "orderby/EPOCH%20desc/"
        f"limit/{count}/"
        "format/tle"
    )


def parse_tle_text(text: str, norad_id: int, name: str):
    """
    Découpe la réponse texte Space-Track (2 lignes par TLE) en une liste
    de tuples (epoch_datetime, name, line1, line2).
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) % 2 != 0:
        raise RuntimeError("Nombre de lignes TLE impair, format inattendu (Space-Track renvoie normalement 2 lignes par TLE).")

    tles = []
    for i in range(0, len(lines), 2):
        l1 = lines[i]
        l2 = lines[i + 1]

        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            print(f"Bloc ignoré (pas TLE valide) :\n{l1}\n{l2}")
            continue

        try:
            catnum = int(l1[2:7])
        except ValueError:
            print(f"Impossible de lire le NORAD dans '{l1}'")
            continue

        if catnum != norad_id:
            continue

        tles.append((parse_tle_epoch(l1[18:32]), name, l1, l2))
    return tles


def fetch_tles_spacetrack(norad_id: int, name: str, count: int):
    """
    Récupère les `count` derniers TLE du satellite `norad_id` via Space-Track,
    triés par epoch décroissant.
    """
    user, password = get_credentials()
    with requests.Session() as s:
        print("Connexion à Space-Track...")
        r_login = s.post(LOGIN_URL, data={"identity": user, "password": password}, timeout=30)
        r_login.raise_for_status()
        # Space-Track renvoie 200 avec un message d'erreur JSON si le login échoue
        if "Failed" in r_login.text:
            raise RuntimeError("Login Space-Track refusé (login/mot de passe ?)")

        print("Connecté, récupération des TLE...")
        r_tle = s.get(tle_query_url(norad_id, count), timeout=30)
        r_tle.raise_for_status()

    text = r_tle.text.strip()
    if not text:
        raise RuntimeError(f"Réponse vide de Space-Track (status {r_tle.status_code}).")

    tles = parse_tle_text(text, norad_id, name)
    if not tles:
        print("Contenu texte Space-Track (début) :")
        print(text[:500])
        raise RuntimeError(f"Aucun TLE {name} ({norad_id}) trouvé dans la réponse Space-Track.")

    tles.sort(key=lambda x: x[0], reverse=True)
    tles = tles[:count]
    print(f"{len(tles)} TLE récupérés pour {name} (NORAD {norad_id}).")
    return tles


def save_tles_to_csv(tles, output_csv: str):
    """
    Enregistre les TLE au format : epoch;name;line1;line2
    """
    print(f"Écriture des TLE dans {output_csv}...")
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["epoch", "name", "line1", "line2"])
        for epoch_dt, name, l1, l2 in tles:
            w.writerow([epoch_dt.isoformat(), name, l1, l2])
    print("Terminé.")


def main():
    parser = argparse.ArgumentParser(description="Télécharge les derniers TLE d'un satellite depuis Space-Track")
    parser.add_argument("--norad", type=int, default=25544, help="NORAD ID (défaut : ISS)")
    parser.add_argument("--name", default="ISS", help="Nom du satellite écrit dans le CSV")
    parser.add_argument("--count", type=int, default=200, help="Nombre de TLE à récupérer")
    parser.add_argument("--output", default=None, help="CSV de sortie (défaut : <name>_last_<count>_tles_spacetrack.csv)")
    args = parser.parse_args()

    output = args.output or f"{args.name.lower()}_last_{args.count}_tles_spacetrack.csv"
    tles = fetch_tles_spacetrack(args.norad, args.name, args.count)
    save_tles_to_csv(tles, output)


if __name__ == "__main__":
    main()
