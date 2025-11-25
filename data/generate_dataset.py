"""
Phase 2: Build multi-satellite dataset using TLE retrieval + SGP4 propagation.
Generates CSV with position/velocity and keplerian elements for requested satellites.
"""
import argparse
import datetime as dt
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from sgp4.api import Satrec, WGS72
from sgp4.api import jday


CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php"


class TLEFetchError(Exception):
    pass


def fetch_tle(identifier: str) -> Tuple[str, str]:
    """Fetch TLE lines from Celestrak using catalog number or common name."""
    params = {"FORMAT": "TLE", "CATNR": identifier}
    # If identifier is not numeric, use NAME param instead
    if not identifier.isdigit():
        params = {"FORMAT": "TLE", "NAME": identifier}
    resp = requests.get(CELESTRAK_URL, params=params, timeout=10)
    if resp.status_code != 200 or len(resp.text.strip().splitlines()) < 2:
        raise TLEFetchError(f"Failed to retrieve TLE for {identifier}: {resp.text}")
    lines = resp.text.strip().splitlines()[:2]
    return lines[0].strip(), lines[1].strip()


def propagate_tle(
    tle: Tuple[str, str], start_time: dt.datetime, end_time: dt.datetime, step_seconds: int
) -> pd.DataFrame:
    """Propagate a TLE between start and end times, returning position/velocity in TEME."""
    l1, l2 = tle
    sat = Satrec.twoline2rv(l1, l2, whichconst=WGS72)

    times: List[dt.datetime] = []
    positions: List[Tuple[float, float, float]] = []
    velocities: List[Tuple[float, float, float]] = []

    current = start_time
    while current <= end_time:
        times.append(current)
        jd, fr = jday(
            current.year, current.month, current.day, current.hour, current.minute, current.second
        )
        error, r, v = sat.sgp4(jd, fr)
        if error != 0:
            raise RuntimeError(f"SGP4 error code {error} at {current}")
        positions.append(tuple(r))
        velocities.append(tuple(v))
        current += dt.timedelta(seconds=step_seconds)

    return pd.DataFrame(
        {
            "epoch_utc": times,
            "x_km": [p[0] for p in positions],
            "y_km": [p[1] for p in positions],
            "z_km": [p[2] for p in positions],
            "vx_km_s": [v[0] for v in velocities],
            "vy_km_s": [v[1] for v in velocities],
            "vz_km_s": [v[2] for v in velocities],
        }
    )


def keplerian_elements_from_tle(tle: Tuple[str, str]) -> Dict[str, float]:
    l1, l2 = tle
    sat = Satrec.twoline2rv(l1, l2, whichconst=WGS72)
    return {
        "mean_motion": sat.no_kozai * 60,  # rad/min -> rad/s
        "incl": sat.inclo,
        "ecc": sat.ecco,
        "raan": sat.nodeo,
        "argp": sat.argpo,
        "M": sat.mo,
    }


def build_dataset(
    satellites: List[str], start_time: dt.datetime, end_time: dt.datetime, step_seconds: int
) -> pd.DataFrame:
    records = []
    for sat_id in satellites:
        try:
            tle = fetch_tle(sat_id)
            print(f"Fetched TLE for {sat_id}")
            kep = keplerian_elements_from_tle(tle)
            df = propagate_tle(tle, start_time, end_time, step_seconds)
            df.insert(0, "satellite", sat_id)
            for k, v in kep.items():
                df[k] = v
            records.append(df)
        except Exception as exc:  # noqa: BLE001 broad to continue other sats
            print(f"[WARN] Skipping {sat_id}: {exc}", file=sys.stderr)
            continue
    if not records:
        raise RuntimeError("No satellites could be propagated. Check identifiers or connectivity.")
    return pd.concat(records, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate multi-satellite propagated dataset")
    parser.add_argument(
        "--sats",
        nargs="+",
        default=["25544"],
        help="List of satellite catalog numbers or names (e.g., 25544 ISS)",
    )
    parser.add_argument("--start", default=None, help="Start time ISO (default: now)")
    parser.add_argument("--end", default=None, help="End time ISO (default: start + 1h)")
    parser.add_argument("--step", type=int, default=60, help="Time step in seconds")
    parser.add_argument("--out", default="multi_sat_dataset.csv", help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    start = dt.datetime.fromisoformat(args.start) if args.start else dt.datetime.utcnow()
    end = dt.datetime.fromisoformat(args.end) if args.end else start + dt.timedelta(hours=1)
    if end <= start:
        raise ValueError("End time must be after start time")
    df = build_dataset(args.sats, start, end, args.step)
    df.to_csv(args.out, index=False)
    print(f"Saved dataset to {args.out} with {len(df)} rows")


if __name__ == "__main__":
    main()