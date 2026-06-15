# Final integrated version
import cv2
import numpy as np
import mediapipe as mp
import speech_recognition as sr
import threading
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
# GLOBAL VARIABLES
# ----------------------------

mode = "VISIBLE"
last_voice_command = "NONE"

# Gesture state tracking
previous_gesture = "NONE"

# ----------------------------
# VOICE THREAD
# ----------------------------

def listen_voice():

    global mode
    global last_voice_command

    recognizer = sr.Recognizer()

    while True:

        try:

            with sr.Microphone() as source:

                audio = recognizer.listen(
                    source,
                    phrase_time_limit=3
                )

            text = recognizer.recognize_google(
                audio
            ).lower()

            print("Detected:", text)

            if "invisible" in text:

                mode = "INVISIBLE"
                last_voice_command = "INVISIBLE"

                print("VOICE -> INVISIBLE")

            elif "visible" in text:

                mode = "VISIBLE"
                last_voice_command = "VISIBLE"

                print("VOICE -> VISIBLE")

        except sr.UnknownValueError:
            pass

        except Exception as e:
            print("Voice Error:", e)

# ----------------------------
# START VOICE THREAD
# ----------------------------

voice_thread = threading.Thread(
    target=listen_voice,
    daemon=True
)

voice_thread.start()

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

while True:

    ret, frame = cap.read()

    if not ret:
        break
    key = cv2.waitKey(1) & 0xFF

    if key == ord('b'):

        background = frame.copy()

        cv2.imwrite(
            "background.jpg",
            background
        )

        print("New background captured!")

    if key == ord('q'):
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

            current_gesture = (
                "OPEN PALM"
                if fingers_up >= 3
                else "CLOSED FIST"
            )

            # Latest gesture change updates mode
            if current_gesture != previous_gesture:

                if current_gesture == "OPEN PALM":
                    mode = "VISIBLE"

                else:
                    mode = "INVISIBLE"

                previous_gesture = current_gesture

            gesture = current_gesture

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

            mask = (mask > 0.5).astype(np.uint8) * 255

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

    cv2.putText(
        output,
        f"VOICE: {last_voice_command}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow(
        "AI Voice + Gesture Invisibility Cloak",
        output
    )


cap.release()
cv2.destroyAllWindows()