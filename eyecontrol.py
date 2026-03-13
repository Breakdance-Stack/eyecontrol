"""
EyeControl - Control your Windows PC with your eyes.
Uses webcam-based iris tracking to replace the mouse cursor.

Key techniques:
- Drift compensation via baseline tracking on raw iris coordinates
- Non-linear sensitivity curve (precise in center, full range at edges)
- Post-saccade settling (averages frames for stable fixation)
"""

import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

try:
    from screeninfo import get_monitors
except ImportError:
    get_monitors = None

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices
# ---------------------------------------------------------------------------
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468

LEFT_EAR_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_INDICES = [33, 160, 158, 133, 153, 144]

CALIBRATION_FILE = Path("calibration_data.json")
LOG_FILE = Path("drift_log.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_screen_size():
    if get_monitors is not None:
        try:
            m = get_monitors()[0]
            return m.width, m.height
        except Exception:
            pass
    return pyautogui.size()


def get_iris_position(landmarks):
    left = landmarks[LEFT_IRIS_CENTER]
    right = landmarks[RIGHT_IRIS_CENTER]
    return (left.x + right.x) / 2.0, (left.y + right.y) / 2.0


def eye_aspect_ratio(landmarks, indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    h_dist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if h_dist == 0:
        return 0.3
    return (v1 + v2) / (2.0 * h_dist)


class MedianFilter:
    def __init__(self, window=7):
        self.buf_x = deque(maxlen=window)
        self.buf_y = deque(maxlen=window)

    def update(self, x, y):
        self.buf_x.append(x)
        self.buf_y.append(y)
        return float(np.median(self.buf_x)), float(np.median(self.buf_y))

    def reset(self):
        self.buf_x.clear()
        self.buf_y.clear()


class DriftCancelSmoother:
    """
    Drift-compensating smoother with:
    - Raw-signal saccade detection (before calibration amplification)
    - Non-linear sensitivity curve (precise center, full range at edges)
    - Post-saccade settling (averages N frames for stable fixation point)
    """

    def __init__(self, screen_w, screen_h,
                 raw_saccade_threshold=0.004,
                 baseline_alpha=0.01,
                 cursor_alpha=0.12,
                 settle_frames=10,
                 curve_exponent=1.8):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.center_x = screen_w / 2.0
        self.center_y = screen_h / 2.0
        self.max_dist = (self.center_x ** 2 + self.center_y ** 2) ** 0.5

        self.raw_saccade_threshold = raw_saccade_threshold
        self.baseline_alpha = baseline_alpha
        self.cursor_alpha = cursor_alpha
        self.settle_frames = settle_frames
        self.curve_exponent = curve_exponent

        self.raw_baseline_x = None
        self.raw_baseline_y = None
        self.target_x = None
        self.target_y = None
        self.cursor_x = None
        self.cursor_y = None

        # Settling state
        self.settling = False
        self.settle_buf = []
        self.settle_remaining = 0

        self.raw_median = MedianFilter(window=7)
        self.mapped_median = MedianFilter(window=7)

    def _apply_curve(self, sx, sy):
        """
        Non-lineare Empfindlichkeitskurve:
        - Kleine Bewegungen (nah an Bildschirmmitte) → stark gedämpft (präzise)
        - Große Bewegungen (Richtung Rand) → weniger gedämpft (volle Reichweite)
        """
        dx = sx - self.center_x
        dy = sy - self.center_y
        dist = (dx * dx + dy * dy) ** 0.5

        if dist < 1.0:
            return self.center_x, self.center_y

        normalized = min(dist / self.max_dist, 1.0)
        scaled = normalized ** self.curve_exponent
        factor = (scaled * self.max_dist) / dist

        out_x = self.center_x + dx * factor
        out_y = self.center_y + dy * factor
        out_x = np.clip(out_x, 5, self.screen_w - 5)
        out_y = np.clip(out_y, 5, self.screen_h - 5)
        return float(out_x), float(out_y)

    def update(self, raw_ix, raw_iy, mapped_sx, mapped_sy):
        filt_rx, filt_ry = self.raw_median.update(raw_ix, raw_iy)
        filt_sx, filt_sy = self.mapped_median.update(mapped_sx, mapped_sy)

        # Nicht-lineare Kurve anwenden
        curved_x, curved_y = self._apply_curve(filt_sx, filt_sy)

        if self.cursor_x is None:
            self.raw_baseline_x = filt_rx
            self.raw_baseline_y = filt_ry
            self.target_x = self.cursor_x = curved_x
            self.target_y = self.cursor_y = curved_y
            return curved_x, curved_y

        # Baseline auf Roh-Signal (absorbiert langsamen Drift)
        self.raw_baseline_x += self.baseline_alpha * (filt_rx - self.raw_baseline_x)
        self.raw_baseline_y += self.baseline_alpha * (filt_ry - self.raw_baseline_y)

        # Saccade-Erkennung auf Roh-Signal
        dev_x = filt_rx - self.raw_baseline_x
        dev_y = filt_ry - self.raw_baseline_y
        dev = (dev_x ** 2 + dev_y ** 2) ** 0.5

        if dev > self.raw_saccade_threshold:
            # Sakkade erkannt → Settling-Phase starten
            self.settling = True
            self.settle_buf = [(curved_x, curved_y)]
            self.settle_remaining = self.settle_frames
            self.raw_baseline_x = filt_rx
            self.raw_baseline_y = filt_ry

        if self.settling:
            self.settle_buf.append((curved_x, curved_y))
            self.settle_remaining -= 1
            if self.settle_remaining <= 0:
                # Settling abgeschlossen: Ziel = Median der gesammelten Frames
                xs = [p[0] for p in self.settle_buf]
                ys = [p[1] for p in self.settle_buf]
                self.target_x = float(np.median(xs))
                self.target_y = float(np.median(ys))
                self.settling = False

        # Cursor gleitet sanft zum Ziel
        self.cursor_x += self.cursor_alpha * (self.target_x - self.cursor_x)
        self.cursor_y += self.cursor_alpha * (self.target_y - self.cursor_y)
        return self.cursor_x, self.cursor_y

    def reset(self):
        self.raw_baseline_x = None
        self.raw_baseline_y = None
        self.target_x = None
        self.target_y = None
        self.cursor_x = None
        self.cursor_y = None
        self.settling = False
        self.settle_buf = []
        self.settle_remaining = 0
        self.raw_median.reset()
        self.mapped_median.reset()


# ---------------------------------------------------------------------------
# Drift Logger
# ---------------------------------------------------------------------------

class DriftLogger:
    def __init__(self, path=LOG_FILE):
        self.path = path
        self.active = False
        self.start_time = 0
        self.duration = 10.0
        self.writer = None
        self.file = None

    def start(self):
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "time_s",
            "raw_iris_x", "raw_iris_y",
            "mapped_screen_x", "mapped_screen_y",
            "smoothed_x", "smoothed_y",
            "delta_from_start_x", "delta_from_start_y",
        ])
        self.active = True
        self.start_time = time.time()
        self.first_smooth_x = None
        self.first_smooth_y = None
        print(f"--- LOG GESTARTET ({self.duration}s) → {self.path} ---")
        print("Schau auf einen festen Punkt und bewege dich nicht!")

    def log(self, raw_ix, raw_iy, mapped_sx, mapped_sy, smooth_x, smooth_y):
        if not self.active:
            return
        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            self.stop()
            return
        if self.first_smooth_x is None:
            self.first_smooth_x = smooth_x
            self.first_smooth_y = smooth_y
        self.writer.writerow([
            f"{elapsed:.3f}",
            f"{raw_ix:.6f}", f"{raw_iy:.6f}",
            f"{mapped_sx:.1f}", f"{mapped_sy:.1f}",
            f"{smooth_x:.1f}", f"{smooth_y:.1f}",
            f"{smooth_x - self.first_smooth_x:.1f}",
            f"{smooth_y - self.first_smooth_y:.1f}",
        ])

    def stop(self):
        if self.file:
            self.file.close()
            self.file = None
        self.active = False
        self.writer = None
        print(f"--- LOG GESTOPPT → {self.path} ---")
        print("Schick mir die Datei drift_log.csv!")

    def is_active(self):
        return self.active


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class Calibrator:
    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.iris_samples = []
        self.screen_points = []
        self.transform_x = None
        self.transform_y = None

    def _target_points(self):
        mx = int(self.screen_w * 0.12)
        my = int(self.screen_h * 0.12)
        return [
            (self.screen_w // 2, self.screen_h // 2),
            (mx, my),
            (self.screen_w - mx, my),
            (mx, self.screen_h - my),
            (self.screen_w - mx, self.screen_h - my),
        ]

    def run(self, face_mesh, cap):
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        points = self._target_points()
        self.iris_samples = []
        self.screen_points = []

        for idx, (sx, sy) in enumerate(points):
            canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            cv2.circle(canvas, (sx, sy), 30, (0, 255, 0), -1)
            cv2.circle(canvas, (sx, sy), 6, (255, 255, 255), -1)
            step_text = f"Schau auf den Punkt ({idx + 1}/{len(points)}) - dann LEERTASTE"
            cv2.putText(canvas, step_text, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, "ESC = Abbrechen", (30, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.imshow("Calibration", canvas)

            collected = []
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    cv2.destroyWindow("Calibration")
                    return False

                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    ix, iy = get_iris_position(lm)
                    collected.append((ix, iy))

                if key == 32 and len(collected) >= 10:
                    recent = collected[-30:]
                    mean_x = np.median([s[0] for s in recent])
                    mean_y = np.median([s[1] for s in recent])
                    self.iris_samples.append((float(mean_x), float(mean_y)))
                    self.screen_points.append((sx, sy))
                    break

        cv2.destroyWindow("Calibration")
        self._fit_transform()
        self._save()
        return True

    def _fit_transform(self):
        iris = np.array(self.iris_samples)
        screen = np.array(self.screen_points)
        if len(iris) >= 5:
            A = np.column_stack([
                np.ones(len(iris)),
                iris[:, 0], iris[:, 1],
                iris[:, 0] * iris[:, 1],
                iris[:, 0] ** 2,
            ])
        else:
            A = np.column_stack([
                np.ones(len(iris)),
                iris[:, 0], iris[:, 1],
            ])
        self.transform_x, _, _, _ = np.linalg.lstsq(A, screen[:, 0], rcond=None)
        self.transform_y, _, _, _ = np.linalg.lstsq(A, screen[:, 1], rcond=None)

    def map(self, ix, iy):
        margin = 5
        if self.transform_x is None:
            sx = np.clip((1.0 - ix) * self.screen_w, margin, self.screen_w - margin)
            sy = np.clip(iy * self.screen_h, margin, self.screen_h - margin)
            return sx, sy
        if len(self.transform_x) == 5:
            features = np.array([1, ix, iy, ix * iy, ix ** 2])
        else:
            features = np.array([1, ix, iy])
        sx = float(features @ self.transform_x)
        sy = float(features @ self.transform_y)
        sx = np.clip(sx, margin, self.screen_w - margin)
        sy = np.clip(sy, margin, self.screen_h - margin)
        return sx, sy

    def _save(self):
        data = {
            "iris_samples": self.iris_samples,
            "screen_points": self.screen_points,
            "transform_x": self.transform_x.tolist(),
            "transform_y": self.transform_y.tolist(),
        }
        CALIBRATION_FILE.write_text(json.dumps(data, indent=2))

    def load(self):
        if not CALIBRATION_FILE.exists():
            return False
        try:
            data = json.loads(CALIBRATION_FILE.read_text())
            self.iris_samples = data["iris_samples"]
            self.screen_points = data["screen_points"]
            self.transform_x = np.array(data["transform_x"])
            self.transform_y = np.array(data["transform_y"])
            return True
        except (json.JSONDecodeError, KeyError):
            return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def safe_move(x, y):
    try:
        pyautogui.moveTo(int(x), int(y), _pause=False)
    except pyautogui.FailSafeException:
        pass


def safe_click(button='left'):
    try:
        pyautogui.click(button=button)
    except pyautogui.FailSafeException:
        pass


def main():
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0

    screen_w, screen_h = get_screen_size()
    print(f"Screen: {screen_w}x{screen_h}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FEHLER: Webcam konnte nicht geöffnet werden.")
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {cam_w}x{cam_h}")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    calibrator = Calibrator(screen_w, screen_h)
    if not calibrator.load():
        print("Keine Kalibrierung gefunden. Starte Kalibrierung...")
        print("Tipp: Bewege nur die Augen, halte den Kopf möglichst still.")
        if not calibrator.run(face_mesh, cap):
            print("Kalibrierung abgebrochen.")
            cap.release()
            sys.exit(0)
        print("Kalibrierung abgeschlossen!")
    else:
        print("Bestehende Kalibrierung geladen. (Taste 'c' zum Neu-Kalibrieren)")

    smoother = DriftCancelSmoother(screen_w, screen_h)
    logger = DriftLogger()

    EAR_THRESHOLD = 0.21
    BLINK_FRAMES = 3
    blink_counter_left = 0
    blink_counter_right = 0
    blink_cooldown = 0

    tracking_active = True
    show_preview = True

    print("\n--- EyeControl aktiv ---")
    print("  Linkes Auge blinzeln  → Linksklick")
    print("  Rechtes Auge blinzeln → Rechtsklick")
    print("  Taste 'c' → Neu-Kalibrieren")
    print("  Taste 't' → Tracking pausieren/fortsetzen")
    print("  Taste 'l' → Drift-Log starten (10s)")
    print("  Taste 'p' → Vorschau ein/aus")
    print("  Taste 'q' / ESC → Beenden")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        sx, sy = 0, 0
        raw_ix, raw_iy = 0, 0
        mapped_sx, mapped_sy = 0, 0

        if results.multi_face_landmarks and tracking_active:
            lm = results.multi_face_landmarks[0].landmark

            raw_ix, raw_iy = get_iris_position(lm)
            mapped_sx, mapped_sy = calibrator.map(raw_ix, raw_iy)
            sx, sy = smoother.update(raw_ix, raw_iy, mapped_sx, mapped_sy)

            safe_move(sx, sy)

            logger.log(raw_ix, raw_iy, mapped_sx, mapped_sy, sx, sy)

            # Blink detection
            if blink_cooldown > 0:
                blink_cooldown -= 1
            else:
                left_ear = eye_aspect_ratio(lm, LEFT_EAR_INDICES, cam_w, cam_h)
                right_ear = eye_aspect_ratio(lm, RIGHT_EAR_INDICES, cam_w, cam_h)

                if left_ear < EAR_THRESHOLD and right_ear >= EAR_THRESHOLD:
                    blink_counter_left += 1
                else:
                    if blink_counter_left >= BLINK_FRAMES:
                        safe_click('left')
                        blink_cooldown = 15
                    blink_counter_left = 0

                if right_ear < EAR_THRESHOLD and left_ear >= EAR_THRESHOLD:
                    blink_counter_right += 1
                else:
                    if blink_counter_right >= BLINK_FRAMES:
                        safe_click('right')
                        blink_cooldown = 15
                    blink_counter_right = 0

        # Preview
        if show_preview:
            display = frame.copy()
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                li = lm[LEFT_IRIS_CENTER]
                ri = lm[RIGHT_IRIS_CENTER]
                cv2.circle(display, (int(li.x * cam_w), int(li.y * cam_h)), 4, (0, 255, 0), -1)
                cv2.circle(display, (int(ri.x * cam_w), int(ri.y * cam_h)), 4, (0, 255, 0), -1)

            status = "AKTIV" if tracking_active else "PAUSIERT"
            if logger.is_active():
                status = "LOG..."
            color = (0, 255, 0) if tracking_active else (0, 0, 255)
            if logger.is_active():
                color = (0, 165, 255)
            cv2.putText(display, f"[{status}] Cursor: ({int(sx)}, {int(sy)})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            small = cv2.resize(display, (320, 240))
            cv2.imshow("EyeControl", small)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            print("Neu-Kalibrierung...")
            calibrator = Calibrator(screen_w, screen_h)
            calibrator.run(face_mesh, cap)
            smoother.reset()
            print("Kalibrierung abgeschlossen!")
        elif key == ord('t'):
            tracking_active = not tracking_active
            state = "aktiv" if tracking_active else "pausiert"
            print(f"Tracking {state}")
            if tracking_active:
                smoother.reset()
        elif key == ord('l'):
            if not logger.is_active():
                logger.start()
            else:
                logger.stop()
        elif key == ord('p'):
            show_preview = not show_preview
            if not show_preview:
                cv2.destroyWindow("EyeControl")

    if logger.is_active():
        logger.stop()
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("EyeControl beendet.")


if __name__ == "__main__":
    main()
