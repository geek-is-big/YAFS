# Simulation Notes

## Run

From project root:

```bash
cd simulation
python3 offloading.py
```

The active scenario is configured in `simulation/offloading.py`:

- `APP_NAME = "CityScenario"`

## Animation Output

Animation is controlled in `simulation/offloading.py`:

- `ENABLE_ANIMATION` - enable/disable animation export
- `ANIMATION_FORMAT` - `"gif"` or `"mp4"`
- `ANIMATION_STEP_STRIDE` - take every N-th simulation step as a frame
- `ANIMATION_MAX_FRAMES` - hard cap for generated frames

Current safe default is:

- `ENABLE_ANIMATION = False`
- `ANIMATION_FORMAT = "mp4"` (can be switched to `"gif"`)

## GIF vs MP4

- GIF export uses Pillow and usually works out of the box, but can be slower/heavier for long runs.
- MP4 export uses Matplotlib `FFMpegWriter` and requires `ffmpeg` in your environment.

If MP4 export fails with writer/codec errors:

1. Verify ffmpeg exists:

```bash
ffmpeg -version
```

2. If missing, install ffmpeg (example with conda):

```bash
conda install -c conda-forge ffmpeg
```

3. Re-run simulation with:

- `ENABLE_ANIMATION = True`
- `ANIMATION_FORMAT = "mp4"`

If ffmpeg is unavailable, set:

- `ANIMATION_FORMAT = "gif"`

or keep animation disabled for fastest simulation.
