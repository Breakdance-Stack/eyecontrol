"""
EyeControl - Control your Windows PC with your eyes.
Uses webcam-based iris tracking to replace the mouse cursor.
"""

import json
import time
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
# Constants – MediaPipe Face Mesh landmark indices
# ---------------------------------------------------------------------------
# Iris centre landmarks (with refine_landmarks=True)
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468

# Eye corner landmarks for computing iris ratio
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

RIGHT_EYE_INNER = 133
RIGHT_EYE_OUTER = 33
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

# Eye Aspect Ratio (EAR) landmarks – 6 points per eye
LEFT_EAR_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_INDICES = [33, 160, 158, 133, 153, 144]

CALIBRATION_FILE = Path("calibration_data.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_screen_size():
    """Return (width, height) of the primary monitor."""
    if get_monitors is not None:
        try:
            m = get_monitors()[0]
            return m.width, m.height
        except Exception:
            pass
    return pyautogui.size()


def eye_aspect_ratio(landmarks, indices, w, h):
    """Compute Eye Aspect Ratio for blink detection."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    # Vertical distances
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # Horizontal distance
    h_dist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if h_dist == 0:
        return 0.3
    return (v1 + v2) / (2.0 * h_dist)


def iris_ratio(landmarks, iris_center_idx, inner_idx, outer_idx, top_idx, bottom_idx):
    """
    Compute the position of the iris centre relative to the eye corners.
    Returns (ratio_x, ratio_y) each in [0, 1].
    """
    iris = landmarks[iris_center_idx]
    inner = landmarks[inner_idx]
    outer = landmarks[outer_idx]
    top = landmarks[top_idx]
    bottom = landmarks[bottom_idx]

    eye_w = abs(inner.x - outer.x)
    eye_h = abs(top.y - bottom.y)

    if eye_w == 0 or eye_h == 0:
        return 0.5, 0.5

    rx = (iris.x - min(inner.x, outer.x)) / eye_w
    ry = (iris.y - min(top.y, bottom.y)) / eye_h

    return np.clip(rx, 0, 1), np.clip(ry, 0, 1)


class Smoother:
    """Exponential moving average for cursor smoothing."""

    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.x = None
        self.y = None

    def update(self, x, y):
        if self.x is None:
            self.x, self.y = x, y
        else:
            self.x = self.alpha * x + (1 - self.alpha) * self.x
            self.y = self.alpha * y + (1 - self.alpha) * self.y
        return self.x, self.y


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class Calibrator:
    """9-point calibration: maps iris ratios to screen coordinates."""

    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.margin = 0.1  # 10 % margin from screen edges
        self.points = self._grid_points()
        self.iris_samples = []  # collected (rx, ry) per point
        self.screen_points = []  # corresponding screen (sx, sy)
        self.transform_x = None  # polynomial coefficients
        self.transform_y = None

    def _grid_points(self):
        """Generate 9 calibration target positions on screen."""
        mx = self.margin * self.screen_w
        my = self.margin * self.screen_h
        cols = [mx, self.screen_w / 2, self.screen_w - mx]
        rows = [my, self.screen_h / 2, self.screen_h - my]
        return [(int(c), int(r)) for r in rows for c in cols]

    def run(self, face_mesh, cap, cam_w, cam_h):
        """
        Run the interactive calibration procedure.
        Shows dots on a full-screen window; user looks at each dot.
        """
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.iris_samples = []
        self.screen_points = []

        for idx, (sx, sy) in enumerate(self.points):
            # Draw target
            canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            cv2.circle(canvas, (sx, sy), 25, (0, 255, 0), -1)
            cv2.circle(canvas, (sx, sy), 5, (255, 255, 255), -1)
            label = f"Look at the green dot ({idx + 1}/{len(self.points)}) - Press SPACE"
            cv2.putText(canvas, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, "Press ESC to cancel", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.imshow("Calibration", canvas)

            # Wait for SPACE, collect samples
            collected = []
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
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
                    lrx, lry = iris_ratio(lm, LEFT_IRIS_CENTER, LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
                    rrx, rry = iris_ratio(lm, RIGHT_IRIS_CENTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)
                    avg_rx = (lrx + rrx) / 2
                    avg_ry = (lry + rry) / 2
                    collected.append((avg_rx, avg_ry))

                if key == 32:  # SPACE
                    if len(collected) >= 5:
                        # Average the last 15 samples for stability
                        recent = collected[-15:]
                        mean_rx = np.mean([s[0] for s in recent])
                        mean_ry = np.mean([s[1] for s in recent])
                        self.iris_samples.append((mean_rx, mean_ry))
                        self.screen_points.append((sx, sy))
                        break

        cv2.destroyWindow("Calibration")
        self._fit_transform()
        self._save()
        return True

    def _fit_transform(self):
        """Fit a 2nd-degree polynomial from iris ratios to screen coords."""
        iris = np.array(self.iris_samples)
        screen = np.array(self.screen_points)

        # Build feature matrix: [1, rx, ry, rx*ry, rx^2, ry^2]
        A = np.column_stack([
            np.ones(len(iris)),
            iris[:, 0], iris[:, 1],
            iris[:, 0] * iris[:, 1],
            iris[:, 0] ** 2, iris[:, 1] ** 2,
        ])

        # Least-squares fit
        self.transform_x, _, _, _ = np.linalg.lstsq(A, screen[:, 0], rcond=None)
        self.transform_y, _, _, _ = np.linalg.lstsq(A, screen[:, 1], rcond=None)

    def map(self, rx, ry):
        """Map iris ratio to screen coordinates using fitted polynomial."""
        if self.transform_x is None:
            # Fallback: simple linear mapping
            return rx * self.screen_w, ry * self.screen_h

        features = np.array([1, rx, ry, rx * ry, rx ** 2, ry ** 2])
        sx = float(features @ self.transform_x)
        sy = float(features @ self.transform_y)
        sx = np.clip(sx, 0, self.screen_w - 1)
        sy = np.clip(sy, 0, self.screen_h - 1)
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
# Main tracking loop
# ---------------------------------------------------------------------------

def main():
    pyautogui.FAILSAFE = True  # move mouse to corner to abort
    pyautogui.PAUSE = 0  # no delay between pyautogui calls

    screen_w, screen_h = get_screen_size()
    print(f"Screen: {screen_w}x{screen_h}")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {cam_w}x{cam_h}")

    # MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Calibration
    calibrator = Calibrator(screen_w, screen_h)
    if not calibrator.load():
        print("No calibration found. Starting calibration...")
        if not calibrator.run(face_mesh, cap, cam_w, cam_h):
            print("Calibration cancelled.")
            cap.release()
            sys.exit(0)
        print("Calibration complete!")
    else:
        print("Loaded existing calibration.")

    smoother = Smoother(alpha=0.25)

    # Blink detection state
    EAR_THRESHOLD = 0.21
    BLINK_FRAMES = 3  # minimum consecutive frames below threshold
    blink_counter_left = 0
    blink_counter_right = 0
    blink_cooldown = 0  # frames to wait after a click

    print("\n--- EyeControl active ---")
    print("Controls:")
    print("  Left eye blink  -> Left click")
    print("  Right eye blink -> Right click")
    print("  Press 'c'       -> Re-calibrate")
    print("  Press 'q'/ESC   -> Quit")
    print("  Move mouse to top-left corner -> Emergency stop (PyAutoGUI failsafe)")

    show_preview = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # --- Gaze tracking ---
            lrx, lry = iris_ratio(lm, LEFT_IRIS_CENTER, LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
            rrx, rry = iris_ratio(lm, RIGHT_IRIS_CENTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)
            avg_rx = (lrx + rrx) / 2
            avg_ry = (lry + rry) / 2

            sx, sy = calibrator.map(avg_rx, avg_ry)
            sx, sy = smoother.update(sx, sy)

            pyautogui.moveTo(int(sx), int(sy), _pause=False)

            # --- Blink detection ---
            if blink_cooldown > 0:
                blink_cooldown -= 1
            else:
                left_ear = eye_aspect_ratio(lm, LEFT_EAR_INDICES, cam_w, cam_h)
                right_ear = eye_aspect_ratio(lm, RIGHT_EAR_INDICES, cam_w, cam_h)

                # Left eye blink (only left eye closed)
                if left_ear < EAR_THRESHOLD and right_ear >= EAR_THRESHOLD:
                    blink_counter_left += 1
                else:
                    if blink_counter_left >= BLINK_FRAMES:
                        pyautogui.click(button='left')
                        blink_cooldown = 15
                    blink_counter_left = 0

                # Right eye blink (only right eye closed)
                if right_ear < EAR_THRESHOLD and left_ear >= EAR_THRESHOLD:
                    blink_counter_right += 1
                else:
                    if blink_counter_right >= BLINK_FRAMES:
                        pyautogui.click(button='right')
                        blink_cooldown = 15
                    blink_counter_right = 0

            # --- Preview window ---
            if show_preview:
                # Draw iris positions on preview
                li = lm[LEFT_IRIS_CENTER]
                ri = lm[RIGHT_IRIS_CENTER]
                cv2.circle(frame, (int(li.x * cam_w), int(li.y * cam_h)), 3, (0, 255, 0), -1)
                cv2.circle(frame, (int(ri.x * cam_w), int(ri.y * cam_h)), 3, (0, 255, 0), -1)
                cv2.putText(frame, f"Cursor: ({int(sx)}, {int(sy)})", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                small = cv2.resize(frame, (320, 240))
                cv2.imshow("EyeControl Preview", small)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            print("Re-calibrating...")
            calibrator.run(face_mesh, cap, cam_w, cam_h)
            smoother = Smoother(alpha=0.25)
            print("Re-calibration complete!")
        elif key == ord('p'):
            show_preview = not show_preview
            if not show_preview:
                cv2.destroyWindow("EyeControl Preview")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("EyeControl stopped.")


if __name__ == "__main__":
    main()
