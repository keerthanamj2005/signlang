import cv2
import numpy as np
import mediapipe as mp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class SignLanguageDetector:
    def __init__(self):
        # Initialize Mediapipe Hands
        self.hands = mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.sentence = ""  # for sentence building

    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmark coordinates as a flattened array."""
        data = []
        for lm in hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
        return np.array(data)

    def dummy_predict(self, landmark_vector):
        """Temporary dummy prediction. Replace with your ML model later."""
        return "HELLO"

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                    # Extract vector
                    landmarks = self.extract_landmarks(handLms)

                    # Predict gesture (dummy)
                    predicted = self.dummy_predict(landmarks)

                    # Add word to sentence
                    if predicted not in self.sentence.split():
                        self.sentence += predicted + " "

            # Display sentence
            cv2.putText(
                frame,
                self.sentence,
                (10, 420),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Sign Language Detection", frame)

            # Quit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
