# --------------------------- PATH FIX (DO NOT REMOVE) ---------------------------
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# --------------------------------------------------------------------------------

import cv2
import joblib
import json
import pyttsx3

from mediapipe_utils.hand_detector import HandDetector
from mediapipe_utils.landmark_extractor import extract_two_hand_landmarks


def speak_macos_safe(text):
    """macOS-safe TTS: reinitialize engine every time"""
    try:
        temp_engine = pyttsx3.init()
        temp_engine.say(text)
        temp_engine.runAndWait()
        temp_engine.stop()
        del temp_engine
    except Exception as e:
        print("TTS Error:", e)


def run_realtime(
    model_path="models/rf_model.joblib",
    labels_path="models/labels.json",
    cam_index=0,
):

    # Load model & labels
    clf = joblib.load(model_path)
    with open(labels_path, "r") as f:
        labels = json.load(f)

    # Setup camera
    cap = cv2.VideoCapture(cam_index)
    detector = HandDetector(max_num_hands=2)

    # ---------------- STABILITY SYSTEM ----------------
    last_pred = None
    stable_count = 0
    STABLE_FRAMES = 2 
    # --------------------------------------------------

    sentence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        # Detect hands
        hands = detector.find_hands(frame, draw=True)

        # Extract landmarks
        vec = extract_two_hand_landmarks(hands, frame.shape)

        if vec is not None:
            try:
                pred_idx = clf.predict([vec])[0]
                pred_label = labels[pred_idx]
                print("RAW:", pred_label)
            except Exception as e:
                cv2.putText(
                    frame,
                    "Model error!",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                print("Prediction error:", e)
                continue

            # ✅ HARD RESET ON GESTURE CHANGE
            if pred_label != last_pred:
                stable_count = 1
            else:
                stable_count += 1

            last_pred = pred_label

            # ✅ SPEAK EXACTLY ONCE PER NEW ACTION
            if stable_count == STABLE_FRAMES:
                print("SPEAK:", pred_label)
                sentence.append(pred_label)

                # ✅ macOS-safe speech
                speak_macos_safe(pred_label)

                confirmed_label = pred_label
            else:
                confirmed_label = ""

            if confirmed_label != "":
                cv2.putText(
                    frame,
                    confirmed_label,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 0, 0),
                    3,
                )
            else:
                cv2.putText(
                    frame,
                    pred_label,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2,
                )

        else:
            cv2.putText(
                frame,
                "Show one or both hands",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # ---------------- SHOW SENTENCE ----------------
        if len(sentence) > 0:
            text = " ".join(sentence[-10:])
            cv2.putText(
                frame,
                text,
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2,
            )

        cv2.imshow("Sign Language Detection (One & Two Hand)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            sentence = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime()
