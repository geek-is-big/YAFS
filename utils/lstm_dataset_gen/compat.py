from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_DIR = REPO_ROOT / "simulation"
SRC_DIR = REPO_ROOT / "src"

_MOBILITY_FUNC: Optional[Callable[[float, float, float, float], float]] = None


def _ensure_import_paths() -> None:
    for path in (SIMULATION_DIR, SRC_DIR, REPO_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_mobility_func() -> Optional[Callable[[float, float, float, float], float]]:
    global _MOBILITY_FUNC
    if _MOBILITY_FUNC is not None:
        return _MOBILITY_FUNC
    _ensure_import_paths()
    try:
        from mobility import haversine_distance_m as mobility_haversine

        _MOBILITY_FUNC = mobility_haversine
    except Exception:
        _MOBILITY_FUNC = None
    return _MOBILITY_FUNC


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    mobility_func = _load_mobility_func()
    if mobility_func is not None:
        return float(mobility_func(lat1, lon1, lat2, lon2))

    radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    return 2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def rssi_from_distance_dbm(distance_m: float) -> float:
    _ensure_import_paths()
    import config as cfg

    if cfg.RSSI_REFERENCE_DISTANCE_M <= 0:
        raise ValueError("Reference distance must be > 0.")
    effective_distance = max(float(distance_m), float(cfg.RSSI_REFERENCE_DISTANCE_M))
    return float(
        cfg.RSSI_AT_REFERENCE_DBM
        - (10.0 * cfg.RSSI_ENVIRONMENT_COEFF * math.log10(effective_distance / cfg.RSSI_REFERENCE_DISTANCE_M))
    )


def rssi_model_parameters() -> tuple[float, float, float]:
    _ensure_import_paths()
    import config as cfg

    return (
        float(cfg.RSSI_REFERENCE_DISTANCE_M),
        float(cfg.RSSI_AT_REFERENCE_DBM),
        float(cfg.RSSI_ENVIRONMENT_COEFF),
    )
