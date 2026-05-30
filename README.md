The emulator was extended to calculate communication energy consumption, making it suitable for mobile-use scenarios.

How to run:
1. Install requirements.txt
2. cd simulation && python offloading.py

You can change simulation scenario by changing the APP_NAME variable in offloading.py

## Black-box Tests

```bash
python tests/blackbox/generate_goldens.py
pytest tests/blackbox -m fast_matrix -q  # use -m full_matrix for the full 2x2x9 suite
```
