#!/usr/bin/env python3
"""
Propagate TLEs with the SGP4 model on a user-defined time grid.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

LOGGER = logging.getLogger(__name__)

COLUMN_ORDER = [
    "sat_id",
    "epoch_utc",
    "datetime_utc",
    "dt_minutes",
    "x_km",
    "y_km",
    "z_km",
    "vx_km_s",
    "vy_km_s",
    "vz_km_s",
]


def configure_logging() -> None:
    """Configure basic console logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Propagate TLEs using SGP4 and export position/velocity time series."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV containing columns sat_id, line1, line2.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV that will contain propagated ephemerides.",
    )
    parser.add_argument(
        "--start-offset-min",
        type=float,
        default=0.0,
        help="Time offset in minutes relative to the TLE epoch at which to start propagation (default: 0).",
    )
    parser.add_argument(
        "--duration-min",
        type=float,
        default=1440.0,
        help="Total propagation duration in minutes (default: 1440 = 1 day).",
    )
    parser.add_argument(
        "--step-min",
        type=float,
        default=10.0,
        help="Propagation step in minutes (default: 10).",
    )
    args = parser.parse_args()

    if args.duration_min < 0:
        parser.error("--duration-min must be non-negative")
    if args.step_min <= 0:
        parser.error("--step-min must be strictly positive")

    return args


def load_tles(csv_path: str) -> pd.DataFrame:
    """
    Load TLEs from a CSV file.

    Expects columns: sat_id (str or int), line1 (str), line2 (str).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_cols = [col for col in ("sat_id", "line1", "line2") if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    return df


def julian_to_datetime(jd: float, fr: float = 0.0) -> datetime:
    """
    Convert Julian date to timezone-aware datetime in UTC.

    Parameters
    ----------
    jd : float
        Julian day integer portion.
    fr : float
        Fractional day.

    Returns
    -------
    datetime
        Datetime in UTC corresponding to jd + fr.
    """
    unix_epoch_jd = 2440587.5  # Julian date at Unix epoch (1970-01-01 00:00:00 UTC)
    total_days = jd + fr - unix_epoch_jd
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(days=total_days)


def time_offsets_minutes(start_offset_min: float, duration_min: float, step_min: float) -> np.ndarray:
    """
    Generate time offsets in minutes starting at start_offset_min for duration_min with spacing step_min.
    """
    num_steps = int(np.floor(duration_min / step_min))
    return start_offset_min + np.arange(num_steps + 1, dtype=float) * step_min


def propagate_tle_row(
    row: pd.Series, start_offset_min: float, duration_min: float, step_min: float
) -> Tuple[pd.DataFrame, int]:
    """
    Propagate a single TLE row over the requested time span.

    Parameters
    ----------
    row : pd.Series
        Row containing sat_id, line1, line2.
    start_offset_min : float
        Start offset from TLE epoch in minutes.
    duration_min : float
        Propagation duration in minutes.
    step_min : float
        Propagation step in minutes.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        DataFrame with propagated states and count of discarded samples due to SGP4 errors.
    """
    sat_id = row["sat_id"]
    line1 = str(row["line1"]).strip()
    line2 = str(row["line2"]).strip()

    try:
        satrec = Satrec.twoline2rv(line1, line2)
    except Exception as exc:
        LOGGER.warning("Failed to parse TLE for sat_id=%s: %s", sat_id, exc)
        return pd.DataFrame(columns=COLUMN_ORDER), 0

    epoch_dt = julian_to_datetime(satrec.jdsatepoch, satrec.jdsatepochF)
    offsets = time_offsets_minutes(start_offset_min, duration_min, step_min)

    records: List[dict] = []
    error_count = 0

    for offset_min in offsets:
        current_dt = epoch_dt + timedelta(minutes=float(offset_min))
        jd, fr = jday(
            current_dt.year,
            current_dt.month,
            current_dt.day,
            current_dt.hour,
            current_dt.minute,
            current_dt.second + current_dt.microsecond * 1e-6,
        )
        error_code, position_km, velocity_km_s = satrec.sgp4(jd, fr)
        if error_code != 0:
            error_count += 1
            LOGGER.debug(
                "SGP4 error %s for sat_id=%s at %s", error_code, sat_id, current_dt.isoformat()
            )
            continue

        records.append(
            {
                "sat_id": sat_id,
                "epoch_utc": epoch_dt.isoformat(),
                "datetime_utc": current_dt.isoformat(),
                "dt_minutes": float(offset_min),
                "x_km": position_km[0],
                "y_km": position_km[1],
                "z_km": position_km[2],
                "vx_km_s": velocity_km_s[0],
                "vy_km_s": velocity_km_s[1],
                "vz_km_s": velocity_km_s[2],
            }
        )

    if not records:
        return pd.DataFrame(columns=COLUMN_ORDER), error_count

    return pd.DataFrame.from_records(records, columns=COLUMN_ORDER), error_count


def propagate_all_tles(
    df: pd.DataFrame, start_offset_min: float, duration_min: float, step_min: float
) -> Tuple[pd.DataFrame, int]:
    """
    Propagate all TLEs in a DataFrame.

    Returns a combined DataFrame sorted by sat_id then datetime_utc,
    along with the count of discarded samples.
    """
    frames: List[pd.DataFrame] = []
    total_errors = 0

    for _, row in df.iterrows():
        row_df, row_errors = propagate_tle_row(row, start_offset_min, duration_min, step_min)
        total_errors += row_errors
        if not row_df.empty:
            frames.append(row_df)

    if frames:
        result = pd.concat(frames, ignore_index=True)
        result.sort_values(by=["sat_id", "datetime_utc"], inplace=True)
    else:
        result = pd.DataFrame(columns=COLUMN_ORDER)

    return result, total_errors


def main() -> None:
    configure_logging()
    args = parse_args()

    LOGGER.info("Loading TLEs from %s", args.input)
    tles_df = load_tles(args.input)

    LOGGER.info(
        "Propagating %d TLEs: start_offset=%s min, duration=%s min, step=%s min",
        len(tles_df),
        args.start_offset_min,
        args.duration_min,
        args.step_min,
    )
    propagated_df, discarded = propagate_all_tles(
        tles_df, args.start_offset_min, args.duration_min, args.step_min
    )

    LOGGER.info("Writing %d propagated points to %s", len(propagated_df), args.output)
    propagated_df.to_csv(args.output, index=False)

    LOGGER.info(
        "Done. Satellites: %d | Samples kept: %d | Discarded due to SGP4 errors: %d | Output: %s",
        len(tles_df),
        len(propagated_df),
        discarded,
        args.output,
    )


if __name__ == "__main__":
    main()
