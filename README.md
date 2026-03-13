# EyeControl

Control your Windows PC with your eyes — no mouse needed. Uses your webcam to track your iris position and translates gaze direction into cursor movement. Blink to click.

## How it works

- **MediaPipe Face Mesh** detects 478 facial landmarks including iris positions in real-time
- **Iris-to-screen mapping** via 9-point calibration with polynomial regression
- **Blink detection** using Eye Aspect Ratio (EAR) — left eye blink = left click, right eye blink = right click
- **Cursor smoothing** with exponential moving average to reduce jitter

## Requirements

- Windows 10/11
- Standard webcam (built-in or USB)
- Good lighting (face a light source for best results)

---

## Option A: Installation auf Windows (PowerShell)

### 1. Python installieren

Öffne **PowerShell als Administrator** (Rechtsklick auf Start > "Windows PowerShell (Administrator)") und führe aus:

```powershell
# Python 3.12 Installer herunterladen
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe" -OutFile "$env:TEMP\python-installer.exe"

# Installer starten (mit PATH-Eintrag und pip)
Start-Process -Wait -FilePath "$env:TEMP\python-installer.exe" -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1"
```

**Wichtig:** Schließe die PowerShell danach und öffne eine **neue PowerShell**, damit Python im PATH ist.

Prüfe ob es geklappt hat:

```powershell
python --version
pip --version
```

### 2. EyeControl herunterladen und starten

```powershell
# Projekt herunterladen und entpacken
Invoke-WebRequest -Uri "https://github.com/Breakdance-Stack/eyecontrol/archive/refs/heads/main.zip" -OutFile "$env:TEMP\eyecontrol.zip"
Expand-Archive -Path "$env:TEMP\eyecontrol.zip" -DestinationPath "$env:USERPROFILE\eyecontrol" -Force

# In den Ordner wechseln
cd "$env:USERPROFILE\eyecontrol\eyecontrol-main"

# Abhängigkeiten installieren
pip install -r requirements.txt

# Starten
python eyecontrol.py
```

---

## Option B: Installation mit WSL2 (Windows Subsystem for Linux)

> **Hinweis:** WSL2 braucht etwas Extra-Setup für Webcam- und GUI-Zugriff. Für die meisten Nutzer ist **Option A (PowerShell)** einfacher.

### 1. WSL2 aktivieren

Öffne **PowerShell als Administrator**:

```powershell
wsl --install -d Ubuntu
```

Starte den PC neu wenn gefordert. Beim ersten Start von Ubuntu legst du einen Benutzernamen und Passwort an.

### 2. Python und Abhängigkeiten in WSL2 installieren

Öffne die **Ubuntu-App** (oder tippe `wsl` in PowerShell):

```bash
# System aktualisieren
sudo apt update && sudo apt upgrade -y

# Python und pip installieren
sudo apt install -y python3 python3-pip python3-venv

# Prüfen
python3 --version
pip3 --version
```

### 3. EyeControl herunterladen und starten

```bash
# Projekt herunterladen und entpacken
curl -L "https://github.com/Breakdance-Stack/eyecontrol/archive/refs/heads/main.tar.gz" -o /tmp/eyecontrol.tar.gz
mkdir -p ~/eyecontrol && tar -xzf /tmp/eyecontrol.tar.gz -C ~/eyecontrol --strip-components=1

# In den Ordner wechseln
cd ~/eyecontrol

# Abhängigkeiten installieren
pip3 install -r requirements.txt

# Starten
python3 eyecontrol.py
```

### 4. WSL2: Webcam und GUI einrichten

Damit EyeControl in WSL2 auf die Webcam und den Bildschirm zugreifen kann:

**GUI-Support (WSLg):** Ab Windows 11 ist WSLg eingebaut — GUI-Fenster funktionieren automatisch. Unter Windows 10 brauchst du einen X-Server wie [VcXsrv](https://sourceforge.net/projects/vcxsrv/):

```bash
# Nur Windows 10 — X-Server Display setzen
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
```

**Webcam in WSL2:** USB-Webcams müssen via [usbipd-win](https://github.com/dorssel/usbipd-win) an WSL2 durchgereicht werden:

```powershell
# In PowerShell (als Administrator):
winget install usbipd

# USB-Geräte auflisten
usbipd list

# Webcam an WSL2 binden (BUSID durch deine Webcam-ID ersetzen)
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

Dann in WSL2:

```bash
# Video-Gerät prüfen
ls /dev/video*

# Nötige Kernel-Module (falls nicht vorhanden)
sudo apt install -y linux-tools-virtual hwdata
sudo modprobe uvcvideo
```

---

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
