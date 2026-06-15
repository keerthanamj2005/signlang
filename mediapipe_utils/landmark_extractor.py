import numpy as np


def extract_landmark_list(hand_landmarks, image_shape):
    """
    Convert one MediaPipe hand_landmarks to a normalized flattened vector (63,)
    """
    h, w = image_shape[:2]
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x * w, lm.y * h, lm.z * w])
    coords = np.array(coords)  # (21,3)

    # translate so wrist (index 0) is origin, then scale by max absolute value
    origin = coords[0]
    coords = coords - origin
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords = coords / max_val
    return coords.flatten()  # shape (63,)


def extract_two_hand_landmarks(hands, image_shape):
    """
    Produces a fixed-length combined vector for 1 or 2 hands:
      - if two hands detected: [hand1(63), hand2(63)]
      - if one hand detected:  [hand1(63), zeros(63)]
      - if no hands: return None
    Note: `hands` is results.multi_hand_landmarks (list-like)
    """
    if not hands:
        return None

    vectors = []
    # take up to 2 hands (MediaPipe returns hands in detection order)
    for hand in hands[:2]:
        vec = extract_landmark_list(hand, image_shape)
        vectors.append(vec)

    # pad if only 1 hand
    if len(vectors) == 1:
        vectors.append(np.zeros(63, dtype=float))

    # if somehow 0 hands (should be caught above)
    if len(vectors) == 0:
        return None

    return np.concatenate(vectors)  # shape (126,)
