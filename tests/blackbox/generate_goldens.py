from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
SIMULATION_PATH = PROJECT_ROOT / "simulation"

for import_path in (str(PROJECT_ROOT), str(SRC_PATH), str(SIMULATION_PATH)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import config as cfg

from tests.blackbox.harness import (
    ALL_CASES,
    case_key,
    run_case,
    snapshot_to_golden_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic black-box golden baselines.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "goldens" / "blackbox_v1.json",
        help="Path to the output JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases_payload = {}
    temp_root = Path(tempfile.mkdtemp(prefix="yafs_blackbox_goldens_"))

    for case in ALL_CASES:
        cfg.STATIC_LINK_RSSI_DBM = float(case.rssi_dbm)
        case_dir = temp_root / case_key(case).replace("|", "__")
        snapshot = run_case(case, output_dir=case_dir)
        cases_payload[case_key(case)] = snapshot_to_golden_payload(snapshot)

    payload = {
        "schema_version": 1,
        "description": (
            "Deterministic black-box regression baselines for simulation/core "
            "transport and energy behavior."
        ),
        "cases": cases_payload,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(cases_payload)} golden cases to {output_path}")


if __name__ == "__main__":
    main()
