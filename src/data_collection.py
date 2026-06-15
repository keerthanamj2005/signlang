# top-of-file helper so this works when run as module or script
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import argparse
from mediapipe_utils.hand_detector import HandDetector
from mediapipe_utils.landmark_extractor import extract_two_hand_landmarks
from src.utils import save_landmark_csv


def collect_data(label, samples=200, cam_index=0):
    print(f"Collecting data for label: {label} (two-hand format: 126 features)")
    print("Press 'q' to stop early.")
    cap = cv2.VideoCapture(cam_index)
    detector = HandDetector(max_num_hands=2)
    count = 0

    while count < samples:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        hands = detector.find_hands(frame, draw=True)  # may be None or list

        vec = extract_two_hand_landmarks(hands, frame.shape)
        if vec is not None:
            # vec is a numpy array (126,)
            save_landmark_csv(label, vec, dataset_dir="dataset")
            count += 1
            cv2.putText(
                frame,
                f"{count}/{samples}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Show one or both hands clearly",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Data Collection (two-hand format)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Data collection completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()
    collect_data(args.label, args.samples, args.cam)
