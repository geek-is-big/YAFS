from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.lstm_dataset_gen.config import load_dataset_config
from utils.lstm_dataset_gen.generator import LSTMDatasetGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic LSTM route and RSSI datasets.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated"),
        help="Directory where routes and RSSI CSV files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed that overrides radio.random_seed from the YAML config.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_dataset_config(args.config)
    generator = LSTMDatasetGenerator(config=config, seed=args.seed)
    datasets = generator.generate_and_save(args.output_dir)

    for name, dataframe in datasets.items():
        print(f"{name}: rows={len(dataframe)}")


if __name__ == "__main__":
    main()
