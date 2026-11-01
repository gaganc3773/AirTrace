"""Offline tests for the real-data clients (OpenAQ, Open-Meteo): parsing, alignment, QC, caching."""

from __future__ import annotations

import numpy as np
import pytest

from airtrace.data.http import write_cache
from airtrace.data.openaq import OpenAQ, hourly_series, pm25_stations, quality_control
from airtrace.data.openmeteo import parse_met


def _hour(utc: str, value: float | None) -> dict:
    return {"value": value, "period": {"datetimeFrom": {"utc": utc, "local": ""}, "datetimeTo": {"utc": "", "local": ""}}}


def test_hourly_series_aligns_on_utc_hours() -> None:
    times = np.arange(np.datetime64("2024-11-01T00"), np.datetime64("2024-11-01T04"))
    res = [_hour("2024-11-01T01:30:00Z", 120.0), _hour("2024-11-01T03:00:00Z", 80.0),
           _hour("2024-11-02T00:00:00Z", 999.0), _hour("2024-11-01T02:00:00Z", None)]
    out = hourly_series(res, times)
    assert np.isnan(out[0]) and out[1] == 120.0 and np.isnan(out[2]) and out[3] == 80.0


def test_quality_control_removes_bad_and_stuck_values() -> None:
    pm = np.array([[50, 51, -3, 2000, 60, 60, 60, 60, 60, 60, 70, 70]], float)
    out = quality_control(pm, max_value=1500, stuck_hours=6)[0]
    assert out[0] == 50 and np.isnan(out[2]) and np.isnan(out[3])
    assert np.isnan(out[4:10]).all()  # six identical hours: frozen analyser
    assert out[10] == 70 and out[11] == 70  # a short repeat is kept


def test_pm25_stations_picks_the_pm25_sensor() -> None:
    locs = [{"id": 7, "name": "Anand Vihar", "provider": {"name": "CPCB"},
             "coordinates": {"latitude": 28.65, "longitude": 77.31},
             "sensors": [{"id": 70, "parameter": {"id": 1}}, {"id": 71, "parameter": {"id": 2}}]},
            {"id": 8, "name": "no pm25", "coordinates": {"latitude": 28.6, "longitude": 77.2},
             "sensors": [{"id": 80, "parameter": {"id": 1}}]}]
    st = pm25_stations(locs)
    assert len(st) == 1 and st[0].sensor_id == 71 and st[0].provider == "CPCB"


def test_openaq_uses_cache_without_key(tmp_path) -> None:
    write_cache(tmp_path / "hours_71_2024-11-01_2024-11-02.json.gz", [_hour("2024-11-01T00:00:00Z", 5.0)])
    client = OpenAQ(None, tmp_path)
    assert client.sensor_hours(71, "2024-11-01", "2024-11-02")[0]["value"] == 5.0
    with pytest.raises(RuntimeError, match="API key"):
        client.sensor_hours(72, "2024-11-01", "2024-11-02")


def test_parse_met_wind_convention_and_mean() -> None:
    raw = [{"hourly": {"time": ["2024-11-01T00:00"], "wind_speed_100m": [2.0], "wind_direction_100m": [270.0],
                       "boundary_layer_height": [100.0]}},
           {"hourly": {"time": ["2024-11-01T00:00"], "wind_speed_100m": [4.0], "wind_direction_100m": [270.0],
                       "boundary_layer_height": [300.0]}}]
    m = parse_met(raw)
    assert np.isclose(m.u[0], 3.0) and abs(m.v[0]) < 1e-9 and m.H[0] == 200.0  # westerly blows towards +x


def _real(T: int = 8):
    from airtrace.data.real import RealData

    times = np.arange(np.datetime64("2025-11-01T00"), np.datetime64("2025-11-01T00") + T)
    pm = np.full((3, T), 100.0)
    pm[1] = 200.0
    pm[2, -1] = np.nan
    return RealData(times, np.array(["a", "b", "c"]), np.array(["x"] * 3), np.zeros((3, 2)),
                    np.array([[5, 5], [5, 5], [10, 20]]), pm, np.full(T, 10.0), np.zeros(T), np.full(T, 40.0))


def test_real_windows_average_shared_cells_and_exclude() -> None:
    from airtrace.data.real import RealWindows
    from airtrace.grid import Grid

    grid = Grid(32, 32, 750.0, 750.0)
    win = RealWindows(_real(), 6, np.zeros((4, 32, 32)), (0.5, 6.0), (175.0, 1560.0))
    v, m = win.rasters(7, grid)
    assert v.shape == (6, 32, 32) and v[0, 5, 5] == 150.0 and m[-1, 10, 20] == 0 and m[0, 10, 20] == 1
    v, m = win.rasters(7, grid, exclude=np.array([True, True, False]))
    assert m[:, 5, 5].sum() == 0 and v[0, 10, 20] == 100.0


def test_real_windows_clip_met_and_local_hour() -> None:
    import torch

    from airtrace.data.real import RealWindows

    data = _real()
    win = RealWindows(data, 6, np.zeros((4, 32, 32)), (0.5, 6.0), (175.0, 1560.0))
    win.set_batch(np.array([7]))
    met = win.met(torch.tensor([0]))
    assert torch.allclose(met.wind_u, torch.full((1, 6), 6.0)) and torch.allclose(met.H, torch.full((1, 6), 175.0))
    assert win.clipped_fraction == {"wind_speed": 1.0, "H": 1.0}
    assert data.local_hour[0] == 7 and int(met.end_hour[0]) == 14  # period 07:30-08:30 UTC ends 14:00 IST
