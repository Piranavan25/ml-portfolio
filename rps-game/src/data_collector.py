# data_collector.py
# Run this script to collect hand landmark data for Rock-Paper-Scissors gestures
import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from collections import deque

# -------------------------------
# MediaPipe setup
# -------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -------------------------------
# Parameters
# -------------------------------
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [2, 6, 10, 14, 18]

# Smoothing deque for stability (optional)
pred_queue = deque(maxlen=8)

# CSV setup
columns = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
df = pd.DataFrame(columns=columns)

# -------------------------------
# Feature extraction
# -------------------------------
def extract_features(hand_landmarks):
    """
    Convert 21 hand landmarks into a 63-length vector: x1..x21, y1..y21, z1..z21
    """
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return features

# -------------------------------
# Main loop
# -------------------------------
cap = cv2.VideoCapture(0)
print("Press 'r' for Rock, 'p' for Paper, 's' for Scissors, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)  # mirror image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand skeleton
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in [ord('r'), ord('p'), ord('s')]:
        if results.multi_hand_landmarks:
            label_map = {ord('r'): 'Rock', ord('p'): 'Paper', ord('s'): 'Scissors'}
            label = label_map[key]
            for hand_landmarks in results.multi_hand_landmarks:
                features = extract_features(hand_landmarks)
                # Convert to DataFrame row
                new_row = pd.DataFrame([features + [label]], columns=df.columns)
                df = pd.concat([df, new_row], ignore_index=True)
            print(f"Captured {label} gesture. Total samples: {len(df)}")
        else:
            print("No hand detected. Try again.")

# -------------------------------
# Save CSV
# -------------------------------
csv_path = os.path.join("..", "data", "rps_hand_landmarks.csv")
df.to_csv(csv_path, index=False)
print(f"All data saved to {csv_path}")

cap.release()
cv2.destroyAllWindows()
hands.close()
