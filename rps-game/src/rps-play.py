# ML-Powered Rock-Paper-Scissors Game
import cv2
import mediapipe as mp
import numpy as np
import random
from collections import deque
import joblib
import time

# 1️ Load Trained Model
clf = joblib.load("../models/rps_gesture_model.pkl")
le = joblib.load("../models/rps_label_encoder.pkl")

GESTURES = le.classes_  # ['Paper', 'Rock', 'Scissors']

# 2️ MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# For smoothing predictions
pred_queue = deque(maxlen=5)

# Finger indices for optional rule-based fallback
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [2, 6, 10, 14, 18]

def extract_features(hand_landmarks):
    """Convert 21 landmarks into 63-length feature vector"""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()

# 3️⃣ Game Logic
def decide_winner(player, computer):
    if player == computer:
        return "Draw"
    elif (player == "Rock" and computer == "Scissors") or \
         (player == "Paper" and computer == "Rock") or \
         (player == "Scissors" and computer == "Paper"):
        return "You Win!"
    else:
        return "Computer Wins!"

# 4️ Webcam & Real-Time Prediction
def main():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    PRED_INTERVAL = 5  # Predict every 5 frames

    player_move = "None"
    computer_move = "None"
    winner_text = ""

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror image
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            frame_count += 1

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw hand skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )

                # Predict gesture every PRED_INTERVAL frames
                if frame_count % PRED_INTERVAL == 0:
                    features = extract_features(hand_landmarks.landmark).reshape(1, -1)
                    pred = le.inverse_transform(clf.predict(features))[0]
                    pred_queue.append(pred)

                    # Smooth prediction
                    player_move = max(set(pred_queue), key=pred_queue.count)

                    # Random computer move and decide winner
                    computer_move = random.choice(GESTURES)
                    winner_text = decide_winner(player_move, computer_move)

                    # Freeze result briefly so player can see
                    time.sleep(1)  # 1 second

            # Display Results
            cv2.putText(frame, f"Player: {player_move}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Computer: {computer_move}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Result: {winner_text}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

            cv2.imshow("Rock-Paper-Scissors", frame)

            # Proper quit: ESC or window close
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or cv2.getWindowProperty("Rock-Paper-Scissors", cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
