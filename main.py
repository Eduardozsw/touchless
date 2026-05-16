import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request, os

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# ── Baixar modelo ──────────────────────────────────────────
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Baixando modelo...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )

# ── Detector ───────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ── Estado ─────────────────────────────────────────────────
pos_buffer     = deque(maxlen=7)
prev_x, prev_y = 0, 0

left_clicking  = False
right_clicking = False
dragging       = False

last_valid_lm   = None
last_valid_time = 0
LAST_VALID_TIMEOUT = 0.3

fps_counter = 0
fps_display = 0
fps_timer   = time.time()

# ── Helpers ────────────────────────────────────────────────
def dist(a, b):
    return np.hypot(a.x - b.x, a.y - b.y)

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def index_pointing(lm):
    tip          = lm[8]
    middle_tip   = lm[12]
    ring_tip     = lm[16]
    pinky_tip    = lm[20]
    others_avg_y = (middle_tip.y + ring_tip.y + pinky_tip.y) / 3
    return tip.y < others_avg_y - 0.04

def detect_gesture(lm):
    index_up  = finger_up(lm, 8, 6)
    middle_up = finger_up(lm, 12, 10)
    ring_up   = finger_up(lm, 16, 14)
    pinky_up  = finger_up(lm, 20, 18)

    pinch_dist   = dist(lm[4], lm[8])
    PINCH_THRESH = 0.05

    # Scroll: punho fechado
    if not index_pointing(lm) and not middle_up and not ring_up and not pinky_up:
        return "SCROLL"

    # Drag: só mínimo levantado
    if pinky_up and not index_up and not middle_up and not ring_up:
        return "DRAG"

    # Clique direito: V + pinça
    if index_up and middle_up and not ring_up and not pinky_up and pinch_dist < PINCH_THRESH:
        return "RIGHT_CLICK"

    # Mover: indicador + médio levantados
    if index_up and middle_up and not ring_up and not pinky_up:
        return "MOVE"

    # Pinça: clique esquerdo
    if pinch_dist < PINCH_THRESH and not middle_up:
        return "PINCH"

    return "NONE"

# ── Mapeamento câmera → tela ───────────────────────────────
SCALE = 2

def map_to_screen(lm):
    raw_x = lm[8].x
    raw_y = lm[8].y

    x = (raw_x - 0.5) * SCALE + 0.5
    y = (raw_y - 0.5) * SCALE + 0.5
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))

    return int(x * SCREEN_W), int(y * SCREEN_H)

# ── Loop principal ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

frame_count = 0
last_result = None

print("Motor de Gestos ativo. ESC para sair.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_count += 1
    if frame_count % 2 == 0:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        last_result = detector.detect(mp_image)

    result = last_result

    # FPS
    fps_counter += 1
    if time.time() - fps_timer >= 1.0:
        fps_display = fps_counter
        fps_counter = 0
        fps_timer   = time.time()

    gesture_label = "---"

    # Última detecção válida com timeout
    if result and result.hand_landmarks:
        lm              = result.hand_landmarks[0]
        last_valid_lm   = lm
        last_valid_time = time.time()
    elif last_valid_lm and (time.time() - last_valid_time) < LAST_VALID_TIMEOUT:
        lm = last_valid_lm
    else:
        lm = None

    if lm:
        gesture       = detect_gesture(lm)
        gesture_label = gesture

        # ── MOVER CURSOR ──────────────────────────────────
        if gesture in ("MOVE", "RIGHT_CLICK"):
            tx, ty = map_to_screen(lm)
            pos_buffer.append((tx, ty))
            sx = int(sum(p[0] for p in pos_buffer) / len(pos_buffer))
            sy = int(sum(p[1] for p in pos_buffer) / len(pos_buffer))
            prev_x, prev_y = sx, sy
            if not dragging:
                pyautogui.moveTo(sx, sy)

        # ── DRAG ──────────────────────────────────────────
        if gesture == "DRAG":
            if not dragging:
                dragging = True
                pyautogui.mouseDown(button="left")
            tx, ty = map_to_screen(lm)
            pos_buffer.append((tx, ty))
            sx = int(sum(p[0] for p in pos_buffer) / len(pos_buffer))
            sy = int(sum(p[1] for p in pos_buffer) / len(pos_buffer))
            pyautogui.moveTo(sx, sy)
            gesture_label = "DRAG"

        # ── PINÇA: clique esquerdo ────────────────────────
        if gesture == "PINCH":
            if not left_clicking:
                left_clicking = True
        else:
            if dragging and gesture != "DRAG":
                pyautogui.mouseUp(button="left")
                dragging = False
            elif left_clicking:
                pyautogui.click(button="left")
                left_clicking = False

        # ── CLIQUE DIREITO ────────────────────────────────
        if gesture == "RIGHT_CLICK" and not right_clicking:
            pyautogui.click(button="right")
            right_clicking = True
        elif gesture != "RIGHT_CLICK":
            right_clicking = False

        # ── SCROLL ────────────────────────────────────────
        if gesture == "SCROLL":
            hand_y       = lm[9].y
            ZONE_TOP     = 0.33
            ZONE_BOTTOM  = 0.66
            SCROLL_SPEED = 20

            if hand_y < ZONE_TOP:
                pyautogui.scroll(SCROLL_SPEED)
                gesture_label = "SCROLL CIMA"
            elif hand_y > ZONE_BOTTOM:
                pyautogui.scroll(-SCROLL_SPEED)
                gesture_label = "SCROLL BAIXO"
            else:
                gesture_label = "SCROLL NEUTRO"

        # ── Desenho dos landmarks ─────────────────────────
        for conn in CONNECTIONS:
            a, b = conn
            ax = int(lm[a].x * w); ay = int(lm[a].y * h)
            bx = int(lm[b].x * w); by = int(lm[b].y * h)
            cv2.line(frame, (ax, ay), (bx, by), (200, 200, 200), 1)
        if lm is not None:
            for p in lm:
                cx, cy = int(p.x * w), int(p.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 100), -1)

    # ── HUD ───────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (320, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesto: {gesture_label}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 100), 2)
    cv2.putText(frame, f"FPS: {fps_display}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

    cv2.imshow("Motor de Gestos - CortechX", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()