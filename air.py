import cv2
import numpy as np
from collections import deque
from pathlib import Path
import time
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


HAND_CONNECTIONS = [
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]


def ensure_model(model_path):
    """Hand Landmarker model yoksa indir."""
    if model_path.exists():
        return

    print(f"Model indiriliyor: {model_path.name}")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Model indirildi.")


ensure_model(MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

detector = vision.HandLandmarker.create_from_options(options)


canvas = None           # çizim katmanı
prev_point = None       # önceki nokta (pürüzsüz çizgi için)

COLORS = [
    (255, 0, 255),      # Magenta
    (0, 0, 255),        # Kırmızı
    (0, 255, 255),      # Sarı
    (0, 255, 0),        # Yeşil
    (255, 0, 0),        # Mavi
    (0, 0, 0),          # Siyah
]
selected_color_index = 2  # Başlangıç: Sarı
draw_color = COLORS[selected_color_index]
draw_thickness = 5
smoothing_buffer = deque(maxlen=5)   # yumuşatma için

# Renk seçim UI konumları
color_positions = {}  # Renklerin ekrandaki konumları


def finger_up(landmarks, tip_id, pip_id):
    """Parmak havada mı? Tip, PIP'ten yukarıdaysa evet."""
    return landmarks[tip_id].y < landmarks[pip_id].y


def count_fingers(hand_landmarks, handedness=None):
    """
    Havadaki parmakları say.
    Döner: (işaret, orta, yüzük, küçük, baş_parmak) - her biri bool
    """
    lm = hand_landmarks

    index  = finger_up(lm, 8,  6)
    middle = finger_up(lm, 12, 10)
    ring   = finger_up(lm, 16, 14)
    pinky  = finger_up(lm, 20, 18)

    # Baş parmak için el yönüne göre x eksenini kontrol ediyoruz.
    is_left_hand = handedness == "Left"
    thumb = lm[4].x > lm[3].x if is_left_hand else lm[4].x < lm[3].x

    return index, middle, ring, pinky, thumb


def smooth_point(x, y):
    """Son birkaç noktanın ortalamasını al → pürüzsüz çizgi."""
    smoothing_buffer.append((x, y))
    avg_x = int(np.mean([p[0] for p in smoothing_buffer]))
    avg_y = int(np.mean([p[1] for p in smoothing_buffer]))
    return avg_x, avg_y


def draw_hand_landmarks(frame, landmarks, connections):
    """El iskeletini OpenCV ile çiz."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start, end in connections:
        cv2.line(frame, points[start], points[end], (80, 80, 80), 2, cv2.LINE_AA)

    for point in points:
        cv2.circle(frame, point, 4, (60, 60, 60), -1, cv2.LINE_AA)


def draw_ui(frame, mode, finger_count):
    """Renk seçim butonları ve bilgi ekle."""
    global color_positions
    h, w = frame.shape[:2]

    radius = 8
    gap = 26
    start_x = int((w - ((len(COLORS) - 1) * gap)) / 2)
    center_y = 26
    color_positions = {}

    for index, color in enumerate(COLORS):
        center_x = start_x + index * gap
        outline = (255, 255, 255) if color == (0, 0, 0) else color
        thickness = -1 if color != (0, 0, 0) else 2
        
        # Seçili renk vurgulanacak
        if index == selected_color_index:
            cv2.circle(frame, (center_x, center_y), radius + 4, (255, 255, 255), 3, cv2.LINE_AA)
        
        cv2.circle(frame, (center_x, center_y), radius, outline, thickness, cv2.LINE_AA)
        color_positions[index] = (center_x, center_y, radius)
    



# Mouse callback fonksiyonu
def mouse_callback(event, x, y, flags, param):
    global selected_color_index, draw_color
    
    if event == cv2.EVENT_LBUTTONDOWN:
        for index, (cx, cy, r) in color_positions.items():
            distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if distance <= r + 5:  # Renk butonuna tıklanıp tıklanmadığını kontrol et
                selected_color_index = index
                draw_color = COLORS[selected_color_index]
                print(f"Renk değiştirildi: {['Magenta', 'Kırmızı', 'Sarı', 'Yeşil', 'Mavi', 'Siyah'][selected_color_index]}")
                break

# ---------------------------------------------
# Ana döngü
# ---------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Air Writing başlatıldı.")
print("  ☝  Sadece işaret parmağı  → ÇİZ")
print("  ✌  İki parmak              → SEÇİM (çizme)")
print("  🖐  Beş parmak              → EKRANI TEMİZLE")
print("  Renkler                    → TIKLAYARAK SEÇ")
print("  Q                          → Çıkış")

mode = "BEKLE"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)          # ayna görüntüsü
    h, w = frame.shape[:2]

    # Canvas başlatma (ilk kare)
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    finger_count = 0

    if results.hand_landmarks:
        hand_lm = results.hand_landmarks[0]

        # İskelet çiz
        draw_hand_landmarks(frame, hand_lm, HAND_CONNECTIONS)

        handedness = None
        if results.handedness:
            handedness = results.handedness[0][0].category_name

        index, middle, ring, pinky, thumb = count_fingers(hand_lm, handedness)
        finger_count = sum([index, middle, ring, pinky, thumb])

        # İşaret parmağı ucu
        tip = hand_lm[8]
        ix, iy = int(tip.x * w), int(tip.y * h)
        sx, sy = smooth_point(ix, iy)

        # -- MOD KARARLARI --
        if finger_count >= 5:
            canvas[:] = 0
            smoothing_buffer.clear()
            prev_point = None
            mode = "TEMİZLENDİ"

        elif index and middle and not ring and not pinky:
            canvas[:] = 0
            prev_point = None
            smoothing_buffer.clear()
            mode = "SİLGİ"

        elif index and not middle:
            mode = "ÇİZİYOR"
            if prev_point is not None:
                cv2.line(canvas, prev_point, (sx, sy), draw_color, draw_thickness, cv2.LINE_AA)
            prev_point = (sx, sy)

            cv2.circle(frame, (ix, iy), 10, draw_color, 2, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), 3, (255, 255, 255), -1, cv2.LINE_AA)

        else:
            prev_point = None
            smoothing_buffer.clear()
            mode = "BEKLE"

    else:
        prev_point = None
        smoothing_buffer.clear()
        mode = "BEKLE"

    # -- CANVAS + FRAME BİRLEŞTİR --
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined = cv2.add(frame_bg, canvas_fg)

    draw_ui(combined, mode, finger_count)

    cv2.imshow("Air Writing", combined)
    cv2.setMouseCallback("Air Writing", mouse_callback)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[:] = 0
        prev_point = None

cap.release()
cv2.destroyAllWindows()
detector.close()
