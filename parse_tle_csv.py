"""Parse TLE lines from a CSV and write expanded fields to a new CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import pandas as pd

def _parse_mantissa_exponent(field: str) -> float:
    """
    Convert a TLE mantissa/exponent string (e.g. ' 29621-4') to a float.
    The format is sign + 5-digit mantissa + sign + 1-digit exponent with
    an implied decimal before the mantissa.
    """
    raw = field.strip()
    if len(raw) == 7:
        sign_char = "+"
        mantissa = raw[0:5]
        exponent = raw[5:]
    elif len(raw) == 8:
        sign_char = raw[0]
        mantissa = raw[1:6]
        exponent = raw[6:]
    else:
        raise ValueError(f"Invalid mantissa/exponent field '{field}'")

    sign = -1.0 if sign_char == "-" else 1.0
    try:
        mantissa_val = float(mantissa)
        exponent_val = int(exponent)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Non-numeric mantissa/exponent in '{field}'") from exc

    return sign * mantissa_val * (10 ** exponent_val) / 1e5


def _parse_epoch_year(two_digit_year: str) -> int:
    """Expand a two-digit TLE year into a four-digit year using the standard 57 pivot."""
    yy = int(two_digit_year)
    return 1900 + yy if yy >= 57 else 2000 + yy


def _parse_eccentricity(ecc_field: str) -> float:
    """Convert the seven-digit TLE eccentricity field to a float."""
    ecc_digits = ecc_field.strip()
    if not ecc_digits or len(ecc_digits) != 7:
        raise ValueError(f"Invalid eccentricity field '{ecc_field}'")
    return float(f"0.{ecc_digits}")


def parse_tle_lines(line1: str, line2: str) -> Dict[str, Any]:
    """
    Parse two TLE lines into a dictionary of NORAD fields.

    Parameters
    ----------
    line1 : str
        The first line of the TLE (must start with '1').
    line2 : str
        The second line of the TLE (must start with '2').

    Returns
    -------
    dict
        Parsed fields mapped by name.
    """
    l1 = line1.rstrip("\r\n")
    l2 = line2.rstrip("\r\n")

    if not l1.startswith("1"):
        raise ValueError(f"Line 1 must start with '1': {line1!r}")
    if not l2.startswith("2"):
        raise ValueError(f"Line 2 must start with '2': {line2!r}")
    if len(l1) < 68 or len(l2) < 68:
        raise ValueError("TLE lines appear too short for fixed-width parsing.")

    try:
        fields: Dict[str, Any] = {
            "satellite_number": int(l1[2:7]),
            "classification": l1[7].strip() or "U",
            "intl_designator_launch_year": int(l1[9:11]),
            "intl_designator_launch_number": int(l1[11:14]),
            "intl_designator_piece": l1[14:17].strip(),
            "epoch_year": _parse_epoch_year(l1[18:20]),
            "epoch_day": float(l1[20:32]),
            "first_derivative_mean_motion": float(l1[33:43]),
            "second_derivative_mean_motion": _parse_mantissa_exponent(l1[44:52]),
            "bstar_drag": _parse_mantissa_exponent(l1[53:61]),
            "ephemeris_type": int(l1[62]),
            "element_set_number": int(l1[64:68]),
            "inclination_deg": float(l2[8:16]),
            "raan_deg": float(l2[17:25]),
            "eccentricity": _parse_eccentricity(l2[26:33]),
            "arg_perigee_deg": float(l2[34:42]),
            "mean_anomaly_deg": float(l2[43:51]),
            "mean_motion_revs_per_day": float(l2[52:63]),
            "revolution_number_at_epoch": int(l2[63:68]),
        }
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Failed to parse TLE lines:\n{line1}\n{line2}") from exc

    return fields


def expand_tle_csv(input_path: Path, output_path: Path) -> None:
    """
    Read an input CSV of TLEs, parse each line pair, and write an expanded CSV.

    The input CSV must contain columns: epoch, name, line1, line2.
    """
    df = pd.read_csv(
        input_path,
        sep =";",
        dtype={"epoch": str, "name": str, "line1": str, "line2": str},
    )

    required_cols = {"epoch", "name", "line1", "line2"}
    missing = required_cols.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")

    records = []
    for _, row in df.iterrows():
        parsed = parse_tle_lines(row["line1"], row["line2"])
        records.append(
            {
                "epoch": row["epoch"],
                "name": row["name"],
                **parsed,
            }
        )

    output_df = pd.DataFrame(records)
    output_df.to_csv(output_path, index=False, encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Parse a CSV containing TLE lines into a new CSV with expanded fields."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Path to the input CSV file.")
    parser.add_argument("output_csv", type=Path, help="Path to write the parsed CSV.")
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.input_csv.exists():
        parser.error(f"Input file does not exist: {args.input_csv}")

    try:
        expand_tle_csv(args.input_csv, args.output_csv)
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    input_csv = Path("/Users/bilaldelais/Desktop/project deep learning/data/raw/iss_last_200000_tles_spacetrack.csv")
    output_csv = Path("/Users/bilaldelais/Desktop/project deep learning/data/processed/iss_200000_parsed.csv")
    
    expand_tle_csv(input_csv, output_csv)


# Example:
#   python parse_tle_csv.py input.csv output.csv
