import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# ----------------------------
# LOAD BACKGROUND
# ----------------------------

background = cv2.imread("background.jpg")

if background is None:
    print("Error: background.jpg not found!")
    exit()

# ----------------------------
# LOAD YOLO MODEL
# ----------------------------

model = YOLO("yolov8m-seg.pt")

# ----------------------------
# MEDIAPIPE HANDS
# ----------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ----------------------------
# CAMERA
# ----------------------------

cap = cv2.VideoCapture(0)

mode = "VISIBLE"

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ----------------------------
    # HAND GESTURE DETECTION
    # ----------------------------

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)

    gesture = "NO HAND"

    if hand_results.multi_hand_landmarks:

        for hand_landmarks in hand_results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            tips = [8, 12, 16, 20]

            fingers_up = 0

            for tip in tips:

                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                    fingers_up += 1

            if fingers_up >= 3:

                gesture = "OPEN PALM"
                mode = "VISIBLE"

            else:

                gesture = "CLOSED FIST"
                mode = "INVISIBLE"

    # ----------------------------
    # YOLO SEGMENTATION
    # ----------------------------

    bg = cv2.resize(
        background,
        (frame.shape[1], frame.shape[0])
    )

    results = model(
    frame,
    imgsz=640,
    verbose=False
)

    human_mask = np.zeros(
        frame.shape[:2],
        dtype=np.uint8
    )

    if results[0].masks is not None:

        classes = results[0].boxes.cls.cpu().numpy()

        for i, mask in enumerate(results[0].masks.data):

            if int(classes[i]) != 0:
                continue

            mask = mask.cpu().numpy()

            mask = cv2.resize(
                mask,
                (frame.shape[1], frame.shape[0])
            )

            mask = (
                mask > 0.5
            ).astype(np.uint8) * 255

            human_mask = cv2.bitwise_or(
                human_mask,
                mask
            )

    # ----------------------------
    # MASK REFINEMENT
    # ----------------------------

    kernel = np.ones((3, 3), np.uint8)

    human_mask = cv2.morphologyEx(
        human_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    human_mask = cv2.morphologyEx(
        human_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    human_mask = cv2.GaussianBlur(
        human_mask,
        (5, 5),
        0
    )

    _, human_mask = cv2.threshold(
        human_mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    inverse_mask = cv2.bitwise_not(
        human_mask
    )

    current_scene = cv2.bitwise_and(
        frame,
        frame,
        mask=inverse_mask
    )

    background_part = cv2.bitwise_and(
        bg,
        bg,
        mask=human_mask
    )

    invisible_frame = cv2.add(
        current_scene,
        background_part
    )

    # ----------------------------
    # DISPLAY
    # ----------------------------

    if mode == "VISIBLE":
        output = frame.copy()
    else:
        output = invisible_frame.copy()

    cv2.putText(
        output,
        f"MODE: {mode}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"GESTURE: {gesture}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Gesture Controlled AI Invisibility",
        output
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()