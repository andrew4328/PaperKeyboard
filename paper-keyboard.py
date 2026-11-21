#!/usr/bin/env python3
"""
Paper + camera keyboard prototype.

  python paper_keyboard.py print --out keyboard.png   # make the printable sheet
  python paper_keyboard.py run                         # preview only (no typing)
  python paper_keyboard.py run --type                  # actually inject keystrokes

Install:
  pip install opencv-contrib-python mediapipe numpy pynput
  (uses the MediaPipe Tasks API; the hand model downloads automatically on first run)

How it works:
  1. 4 ArUco markers at the sheet corners -> homography (image px -> sheet coords)
  2. MediaPipe Hands -> 5 fingertips per hand
  3. Tap = fingertip moves DOWN in the image quickly, then stops (motion signature)
  4. Fingertip position at the stop, mapped to sheet coords, picks the key
"""
import argparse
import collections
import time

import cv2
import numpy as np

# ----------------------------------------------------------------- layout --
CANVAS_W, CANVAS_H = 1000, 560          # sheet coordinate system (arbitrary units)
MARKER_SIZE, MARKER_MARGIN = 90, 15
MARKER_IDS = [0, 1, 2, 3]               # order: top-left, top-right, bottom-right, bottom-left
ARUCO_DICT = cv2.aruco.DICT_4X4_50

UNIT, ROW_H, X0, Y0 = 85, 66, 75, 115   # key width unit, row height, keyboard origin
WIDTHS = {"BKSP": 2, "SPACE": 5, "ENTER": 2}
ROWS = [
    (0.0, "1 2 3 4 5 6 7 8 9 0".split()),
    (0.0, list("QWERTYUIOP")),
    (0.5, list("ASDFGHJKL")),
    (0.5, list("ZXCVBNM,.")),
    (0.5, ["BKSP", "SPACE", "ENTER"]),
]

_c = MARKER_MARGIN + MARKER_SIZE / 2
MARKER_CENTERS = {
    0: (_c, _c),
    1: (CANVAS_W - _c, _c),
    2: (CANVAS_W - _c, CANVAS_H - _c),
    3: (_c, CANVAS_H - _c),
}

TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def build_keys():
    keys = []
    for r, (offset, labels) in enumerate(ROWS):
        x = X0 + offset * UNIT
        y = Y0 + r * ROW_H
        for label in labels:
            w = WIDTHS.get(label, 1) * UNIT
            keys.append({"label": label, "rect": (x, y, x + w, y + ROW_H)})
            x += w
    return keys


# ------------------------------------------------------------ sheet maker --
def generate_sheet(path, scale=3):
    img = np.full((CANVAS_H * scale, CANVAS_W * scale), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 3 * scale

    for k in build_keys():
        x0, y0, x1, y1 = [int(v * scale) for v in k["rect"]]
        cv2.rectangle(img, (x0 + pad, y0 + pad), (x1 - pad, y1 - pad), 0, max(2, scale))
        label = k["label"]
        fs = (0.6 if len(label) > 1 else 0.9) * scale
        th = max(2, scale)
        (tw, tht), _ = cv2.getTextSize(label, font, fs, th)
        cv2.putText(img, label, ((x0 + x1 - tw) // 2, (y0 + y1 + tht) // 2),
                    font, fs, 0, th, cv2.LINE_AA)

    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    size = MARKER_SIZE * scale
    for mid, (cx, cy) in MARKER_CENTERS.items():
        m = cv2.aruco.generateImageMarker(d, mid, size)
        x, y = int(cx * scale - size / 2), int(cy * scale - size / 2)
        img[y:y + size, x:x + size] = m

    cv2.imwrite(path, img)
    print(f"Wrote {path}. Print landscape at 100% / fit-to-width, keep aspect ratio, "
          f"matte paper (glossy glare hurts marker detection).")


# ------------------------------------------------------------ tap detector --
class TapDetector:
    """Tap = fast downward fingertip motion (image y increasing) followed by a stop."""

    def __init__(self, down, stop, cooldown=0.30, window=0.25):
        self.down, self.stop, self.cooldown, self.window = down, stop, cooldown, window
        self.hist, self.last_tap = {}, {}

    def update(self, key, t, y, scale):
        d = self.hist.setdefault(key, collections.deque())
        d.append((t, y))
        while d and t - d[0][0] > self.window:
            d.popleft()
        if len(d) < 3:
            return False, 0.0
        # velocities in hand-lengths per second (scale-invariant)
        v = [(d[i][1] - d[i - 1][1]) / max(d[i][0] - d[i - 1][0], 1e-3) / scale
             for i in range(1, len(d))]
        peak, last = max(v[:-1]), v[-1]
        if peak > self.down and last < self.stop and t - self.last_tap.get(key, -9) > self.cooldown:
            self.last_tap[key] = t
            d.clear()
            return True, peak
        return False, peak


# --------------------------------------------------------------- main loop --
def key_at(keys, pt):
    x, y = pt
    for k in keys:
        x0, y0, x1, y1 = k["rect"]
        if x0 <= x < x1 and y0 <= y < y1:
            return k["label"]
    return None


def send_key(ctrl, label):
    from pynput.keyboard import Key
    special = {"SPACE": Key.space, "ENTER": Key.enter, "BKSP": Key.backspace}
    k = special.get(label, label.lower())
    ctrl.press(k)
    ctrl.release(k)


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = "hand_landmarker.task"


def ensure_model():
    """The Tasks API needs a model file; download it once next to the script."""
    import os
    import urllib.request
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand model to {MODEL_PATH} ...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            raise SystemExit(f"Could not download the model ({e}).\n"
                             f"Download it manually from:\n  {MODEL_URL}\n"
                             f"and save it as {MODEL_PATH} in this folder.")
    return MODEL_PATH


def run(args):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    keys = build_keys()
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters())
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5))
    last_ts = 0
    taps = TapDetector(args.down, args.stop)

    ctrl = None
    if args.type:
        from pynput.keyboard import Controller
        ctrl = Controller()

    import sys
    cam = int(args.camera) if str(args.camera).isdigit() else args.camera
    if isinstance(cam, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera!r}. Try another index "
                         f"(--camera 1) or a stream URL (--camera http://phone-ip:8080/video).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    H = H_inv = None
    text, flash, debug = "", {}, args.debug
    dst = np.float32([MARKER_CENTERS[m] for m in MARKER_IDS])
    print("Running. 'q' = quit, 'd' = toggle debug speeds.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        h, w = frame.shape[:2]

        # 1. Find the paper
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is not None:
            found = {int(i): c[0].mean(axis=0) for i, c in zip(ids.flatten(), corners)}
            if all(m in found for m in MARKER_IDS):
                src = np.float32([found[m] for m in MARKER_IDS])
                Hn, _ = cv2.findHomography(src, dst)
                if Hn is not None:
                    Hn /= Hn[2, 2]
                    H = Hn if H is None else 0.7 * H + 0.3 * Hn   # light smoothing
                    H_inv = np.linalg.inv(H)

        if H is None:
            cv2.putText(frame, "Looking for the 4 corner markers...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            for k in keys:  # draw key outlines projected onto the camera image
                x0, y0, x1, y1 = k["rect"]
                quad = np.float32([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]])
                pts = cv2.perspectiveTransform(quad, H_inv).astype(np.int32)
                hot = now - flash.get(k["label"], -9) < 0.2
                cv2.polylines(frame, [pts], True, (0, 255, 0) if hot else (255, 180, 0),
                              3 if hot else 1)

            # 2. Track fingertips
            last_ts = max(last_ts + 1, int(now * 1000))   # must strictly increase
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = hands.detect_for_video(mp_img, last_ts)
            if res.hand_landmarks:
                for lms, hd in zip(res.hand_landmarks, res.handedness):
                    side = hd[0].category_name
                    p = np.array([[l.x * w, l.y * h] for l in lms])
                    scale = np.linalg.norm(p[0] - p[9]) + 1e-6   # wrist -> middle knuckle

                    for name, idx in TIPS.items():
                        tip = p[idx]
                        cv2.circle(frame, tuple(tip.astype(int)), 5, (0, 255, 255), -1)

                        # 3. Tap?
                        tapped, peak = taps.update((side, name), now, tip[1], scale)
                        if debug:
                            cv2.putText(frame, f"{peak:.1f}", tuple((tip + [8, -8]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        if not tapped:
                            continue

                        # 4. Which key?
                        sheet_pt = cv2.perspectiveTransform(
                            tip.reshape(1, 1, 2).astype(np.float32), H)[0, 0]
                        label = key_at(keys, sheet_pt)
                        if label is None:
                            continue
                        flash[label] = now
                        print(f"{side} {name} -> {label}")
                        if ctrl:
                            send_key(ctrl, label)
                        if label == "BKSP":
                            text = text[:-1]
                        elif label == "SPACE":
                            text += " "
                        elif label == "ENTER":
                            text += "|"
                        else:
                            text += label.lower()

        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, text[-60:] + "_", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
        cv2.imshow("paper keyboard", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("d"):
            debug = not debug

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("print", help="generate the printable keyboard sheet")
    p.add_argument("--out", default="keyboard.png")

    r = sub.add_parser("run", help="start the camera keyboard")
    r.add_argument("--camera", default="0", help="camera index (0, 1, ...) or a stream URL")
    r.add_argument("--type", action="store_true", help="inject real keystrokes (default: preview only)")
    r.add_argument("--down", type=float, default=2.5, help="min downward speed to count as a tap")
    r.add_argument("--stop", type=float, default=0.8, help="max speed after the stroke (finger stopped)")
    r.add_argument("--debug", action="store_true", help="show per-finger speeds")

    args = ap.parse_args()
    generate_sheet(args.out) if args.cmd == "print" else run(args)


if __name__ == "__main__":
    main()
