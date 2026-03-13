"""
EyeControl - Control your Windows PC with your eyes.
Uses webcam-based iris tracking to replace the mouse cursor.

Tracking uses iris position in the full camera frame (captures both
head movement and eye movement) for reliable full-screen coverage.
"""

import json
import sys
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

# EAR landmarks – 6 points per eye
LEFT_EAR_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_INDICES = [33, 160, 158, 133, 153, 144]

CALIBRATION_FILE = Path("calibration_data.json")


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
    """
    Get the average iris center position in normalized frame coordinates (0-1).
    This captures both head movement and eye movement for maximum range.
    """
    left = landmarks[LEFT_IRIS_CENTER]
    right = landmarks[RIGHT_IRIS_CENTER]
    x = (left.x + right.x) / 2.0
    y = (left.y + right.y) / 2.0
    return x, y


def eye_aspect_ratio(landmarks, indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    h_dist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if h_dist == 0:
        return 0.3
    return (v1 + v2) / (2.0 * h_dist)


class Smoother:
    """
    Adaptive smoother with deadzone to eliminate stick-drift.
    Small movements below the deadzone threshold are completely ignored.
    """

    def __init__(self, alpha_slow=0.04, alpha_fast=0.35,
                 speed_threshold=50.0, deadzone=8.0):
        self.alpha_slow = alpha_slow
        self.alpha_fast = alpha_fast
        self.speed_threshold = speed_threshold
        self.deadzone = deadzone
        self.x = None
        self.y = None

    def update(self, x, y):
        if self.x is None:
            self.x, self.y = x, y
            return self.x, self.y

        speed = ((x - self.x) ** 2 + (y - self.y) ** 2) ** 0.5

        # Deadzone: ignoriere Mikro-Bewegungen (Rauschen / Drift)
        if speed < self.deadzone:
            return self.x, self.y

        t = min(speed / self.speed_threshold, 1.0)
        alpha = self.alpha_slow + t * (self.alpha_fast - self.alpha_slow)

        self.x = alpha * x + (1 - alpha) * self.x
        self.y = alpha * y + (1 - alpha) * self.y
        return self.x, self.y

    def reset(self):
        self.x = None
        self.y = None


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class Calibrator:
    """
    5-point calibration: maps iris frame-coordinates to screen coordinates.
    Uses the iris position in the full camera frame (not relative to eye corners)
    for much better range and stability.
    """

    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.iris_samples = []
        self.screen_points = []
        self.transform_x = None
        self.transform_y = None

    def _target_points(self):
        """5 calibration points: center + 4 corners with margin."""
        mx = int(self.screen_w * 0.12)
        my = int(self.screen_h * 0.12)
        return [
            (self.screen_w // 2, self.screen_h // 2),  # center first
            (mx, my),                                    # top-left
            (self.screen_w - mx, my),                    # top-right
            (mx, self.screen_h - my),                    # bottom-left
            (self.screen_w - mx, self.screen_h - my),    # bottom-right
        ]

    def run(self, face_mesh, cap):
        """Interactive calibration: show dots, user looks at each, press SPACE."""
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        points = self._target_points()
        self.iris_samples = []
        self.screen_points = []

        for idx, (sx, sy) in enumerate(points):
            # Draw target
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

                if key == 32 and len(collected) >= 10:  # SPACE
                    recent = collected[-20:]
                    mean_x = np.mean([s[0] for s in recent])
                    mean_y = np.mean([s[1] for s in recent])
                    self.iris_samples.append((mean_x, mean_y))
                    self.screen_points.append((sx, sy))
                    break

        cv2.destroyWindow("Calibration")
        self._fit_transform()
        self._save()
        return True

    def _fit_transform(self):
        """Fit affine transform from iris coords to screen coords."""
        iris = np.array(self.iris_samples)
        screen = np.array(self.screen_points)

        # Feature matrix: [1, x, y, x*y, x^2, y^2] if enough points,
        # else simpler [1, x, y]
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
        """Map iris frame position to screen coordinates."""
        if self.transform_x is None:
            margin = 5
            sx = np.clip((1.0 - ix) * self.screen_w, margin, self.screen_w - margin)
            sy = np.clip(iy * self.screen_h, margin, self.screen_h - margin)
            return sx, sy

        if len(self.transform_x) == 5:
            features = np.array([1, ix, iy, ix * iy, ix ** 2])
        else:
            features = np.array([1, ix, iy])

        sx = float(features @ self.transform_x)
        sy = float(features @ self.transform_y)
        # Sicherheitsrand: nie in die Ecken (verhindert PyAutoGUI Failsafe)
        margin = 5
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
    """Move cursor with failsafe protection — never crash, just skip."""
    try:
        pyautogui.moveTo(int(x), int(y), _pause=False)
    except pyautogui.FailSafeException:
        pass  # Ignorieren, nächster Frame korrigiert die Position


def safe_click(button='left'):
    """Click with failsafe protection."""
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

    # --- Calibration ---
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

    smoother = Smoother()

    # Blink state
    EAR_THRESHOLD = 0.21
    BLINK_FRAMES = 3
    blink_counter_left = 0
    blink_counter_right = 0
    blink_cooldown = 0

    tracking_active = True
    show_preview = True

    print("\n--- EyeControl aktiv ---")
    print("  Linkes Auge blinzeln  -> Linksklick")
    print("  Rechtes Auge blinzeln -> Rechtsklick")
    print("  Taste 'c' -> Neu-Kalibrieren")
    print("  Taste 't' -> Tracking pausieren/fortsetzen")
    print("  Taste 'p' -> Vorschau ein/aus")
    print("  Taste 'q' / ESC -> Beenden")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        sx, sy = 0, 0

        if results.multi_face_landmarks and tracking_active:
            lm = results.multi_face_landmarks[0].landmark

            # --- Gaze ---
            ix, iy = get_iris_position(lm)
            sx, sy = calibrator.map(ix, iy)
            sx, sy = smoother.update(sx, sy)

            safe_move(sx, sy)

            # --- Blink detection ---
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

        # --- Preview ---
        if show_preview:
            display = frame.copy()
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                li = lm[LEFT_IRIS_CENTER]
                ri = lm[RIGHT_IRIS_CENTER]
                cv2.circle(display, (int(li.x * cam_w), int(li.y * cam_h)), 4, (0, 255, 0), -1)
                cv2.circle(display, (int(ri.x * cam_w), int(ri.y * cam_h)), 4, (0, 255, 0), -1)

            status = "AKTIV" if tracking_active else "PAUSIERT"
            color = (0, 255, 0) if tracking_active else (0, 0, 255)
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
        elif key == ord('p'):
            show_preview = not show_preview
            if not show_preview:
                cv2.destroyWindow("EyeControl")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("EyeControl beendet.")


if __name__ == "__main__":
    main()
