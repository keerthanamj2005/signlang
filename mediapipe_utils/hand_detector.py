import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class HandDetector:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        # default max_num_hands=2 to support one- and two-hand gestures
        self.hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def find_hands(self, image, draw=True):
        """
        image: BGR image (OpenCV)
        returns: results.multi_hand_landmarks (or None)
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        hands_landmarks = results.multi_hand_landmarks
        if draw and hands_landmarks:
            for handLms in hands_landmarks:
                mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
        return hands_landmarks

    def close(self):
        self.hands.close()
