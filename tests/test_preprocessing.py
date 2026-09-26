import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preproccessing"))

from fetch_tles import parse_tle_epoch, parse_tle_text  # noqa: E402
from fix_time_truncation import satellite_from_params  # noqa: E402
from sgp4_TLE_horizons import generate_sgp4_trajectory, to_skyfield_time  # noqa: E402

ISS_L1 = "1 25544U 98067A   25335.89873625  .00009722  00000-0  18290-3 0  9999"
ISS_L2 = "2 25544  51.6307 196.1486 0003628 192.4733 167.6166 15.49234272541177"


def test_parse_tle_epoch():
    dt = parse_tle_epoch("25335.89873625")
    expected = datetime(2025, 12, 1, tzinfo=timezone.utc) + timedelta(days=0.89873625)
    assert abs((dt - expected).total_seconds()) < 1e-3


def test_parse_tle_epoch_pre_2000():
    assert parse_tle_epoch("98001.00000000").year == 1998


def test_parse_tle_text_filters_other_norad():
    other = ISS_L1.replace("25544", "20580"), ISS_L2.replace("25544", "20580")
    text = "\n".join([ISS_L1, ISS_L2, *other])
    tles = parse_tle_text(text, 25544, "ISS")
    assert len(tles) == 1 and tles[0][2] == ISS_L1


def test_skyfield_time_keeps_microseconds():
    dt = datetime(2025, 12, 1, 21, 34, 10, 812000, tzinfo=timezone.utc)
    t = to_skyfield_time(dt)
    assert abs((t.utc_datetime() - dt).total_seconds()) < 1e-5


def test_sgp4_at_tle_epoch_matches_exact_time():
    """Le 1er point d'un TLE (dt=0) doit être propagé à l'epoch exacte (fraction de seconde incluse)."""
    epoch = parse_tle_epoch(ISS_L1[18:32])
    assert epoch.microsecond != 0
    times, pos, _, dt_since, _ = generate_sgp4_trajectory([(epoch, "ISS", ISS_L1, ISS_L2)], extra_hours_last=1)
    assert dt_since[0] == 0.0
    from sgp4_TLE_horizons import ts
    from skyfield.api import EarthSatellite
    exact = EarthSatellite(ISS_L1, ISS_L2, "ISS", ts).at(ts.from_datetime(epoch)).position.km
    assert np.linalg.norm(pos[0] - exact) < 1e-6


@pytest.mark.parametrize("path", ["data/datasetISS_200TLE.csv", "data/dataset_hst_sgp4_vs_horizons2.csv"])
def test_datasets_sgp4_columns_use_exact_time(path):
    """Les positions SGP4 stockées correspondent à l'instant exact (pas tronqué à la seconde)."""
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / path, sep=";")
    for tle_index in df["tle_index"].unique()[:5]:
        row = df[df["tle_index"] == tle_index].iloc[0]
        sat = satellite_from_params(row)
        t = pd.Timestamp(row["time_utc"]).to_pydatetime()
        expected = sat.at(to_skyfield_time(t)).position.km
        stored = row[["x_sgp4_km", "y_sgp4_km", "z_sgp4_km"]].to_numpy(dtype=float)
        assert np.linalg.norm(stored - expected) < 1e-3
