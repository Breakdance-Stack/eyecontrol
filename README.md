# EyeControl

Control your Windows PC with your eyes — no mouse needed. Uses your webcam to track your iris position and translates gaze direction into cursor movement. Blink to click.

## How it works

- **MediaPipe Face Mesh** detects 478 facial landmarks including iris positions in real-time
- **Iris-to-screen mapping** via 9-point calibration with polynomial regression
- **Blink detection** using Eye Aspect Ratio (EAR) — left eye blink = left click, right eye blink = right click
- **Cursor smoothing** with exponential moving average to reduce jitter

## Requirements

- Windows 10/11
- Python 3.9+
- Standard webcam (built-in or USB)
- Good lighting (face a light source for best results)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/eyecontrol.git
cd eyecontrol
pip install -r requirements.txt
```

## Usage

```bash
python eyecontrol.py
```

### First launch

On first launch, a **calibration screen** appears with 9 green dots. Look at each dot and press **SPACE** to confirm. This takes about 30 seconds.

Calibration data is saved to `calibration_data.json` and reused on subsequent launches.

### Controls

| Action | Effect |
|--------|--------|
| Look around | Move cursor |
| Left eye blink (hold ~300ms) | Left click |
| Right eye blink (hold ~300ms) | Right click |
| Press `c` | Re-calibrate |
| Press `p` | Toggle preview window |
| Press `q` or `ESC` | Quit |
| Move mouse to top-left corner | Emergency stop (PyAutoGUI failsafe) |

## Tips

- **Sit still** during calibration — head movement reduces accuracy
- **Face a light source** — poor lighting degrades iris tracking
- **Re-calibrate** (`c`) if you move your head or chair significantly
- The system works best for **clicking large UI elements** (icons, buttons, menus). It is not pixel-precise
- Glasses are supported but thick frames or strong reflections may reduce accuracy

## Configuration

Edit the constants at the top of `eyecontrol.py` to tune behavior:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EAR_THRESHOLD` | `0.21` | Eye Aspect Ratio threshold for blink detection |
| `BLINK_FRAMES` | `3` | Minimum consecutive frames for a blink to register |
| `Smoother(alpha=)` | `0.25` | Cursor smoothing factor (lower = smoother but laggier) |

## Tech Stack

- [MediaPipe](https://mediapipe.dev/) — Face Mesh with iris landmarks
- [OpenCV](https://opencv.org/) — Webcam capture and preview
- [PyAutoGUI](https://pyautogui.readthedocs.io/) — Cursor movement and click simulation
- [NumPy](https://numpy.org/) — Calibration math and smoothing

## License

MIT
