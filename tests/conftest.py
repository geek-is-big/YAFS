from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
SIMULATION_PATH = PROJECT_ROOT / "simulation"

for import_path in (str(SRC_PATH), str(SIMULATION_PATH)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)
