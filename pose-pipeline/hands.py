import mediapipe as mp
import cv2
import os
import numpy as np

# This uses a legacy version of a mediapipe solution
# The documentation for the legacy version can be found here: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
# Support for it has ended on March 1st, 2023: https://ai.google.dev/edge/mediapipe/solutions/guide#legacy 

__hands = mp.solutions.hands.Hands(
    static_image_mode = os.environ.get("STATIC_IMAGE_MODE", False),
    max_num_hands = os.environ.get("MAX_NUM_HANDS", 1),
    min_detection_confidence=os.environ.get("MIN_DETECTION_CONFIDENCE", 0.5),
    min_tracking_confidence=os.environ.get("MIN_TRACKING_CONFIDENCE", 0.5)
)

def run(frame, colorspace_convert=cv2.COLOR_BGR2RGB, **kwargs):
    if 'flipH' in kwargs and kwargs['flipH']:
        cv2.flip(frame, 1)
    if 'flipV' in kwargs and kwargs['flipV']:
        cv2.flip(frame, 0)

    if colorspace_convert:
        frame = cv2.cvtColor(frame, colorspace_convert)
    return __hands.process(frame)

# TODO - normalize handedness
'''
def post_process(results, keypoints=mp.solutions.hands.HandLandmark, world=True, coords="xyz", normalize_handedness = "left", **kwargs):
    lm_list = results.multi_hand_landmarks
    if world:
        lm_list = results.multi_hand_world_landmarks
    if lm_list:
        #TODO: Fix this list comprehension abomination
        # It iterates through all frames, keypoints, and xyz values, then flips the x value
        return np.array([[[
            1 - getattr(lm.landmark[keypoint], coord) 
                if results.multi_handedness[idx].classification[0].label.lower() == "left" and coord == "x" and not world
            else getattr(lm.landmark[keypoint], coord) 
            for coord in coords
            ] for keypoint in keypoints] for idx, lm in enumerate(lm_list)])

    else:
        return np.zeros((1, 21, len(coords)))
'''

def post_process(results, keypoints=mp.solutions.hands.HandLandmark, world=False, coords="xyz"):
    """
    Processes the hand landmarks from MediaPipe results.

    Parameters:
    - results: The output from MediaPipe hand tracking.
    - keypoints: The specific hand keypoints to extract (default: all 21 hand landmarks).
    - world (bool): If True, use world coordinates (3D). If False, use image coordinates (2D).
    - coords (str): Specifies which coordinate dimensions to extract (e.g., "xyz" or "xy").

    Returns:
    - A NumPy array of shape (num_hands, 21, len(coords)) containing extracted hand landmarks.
      If no hands are detected, returns an array of zeros.
    """
    # Select the appropriate landmark list (world or image coordinates)
    lm_list = results.multi_hand_world_landmarks if world else results.multi_hand_landmarks

    # If no hands are detected, return a zero array with the expected shape
    if not lm_list:
        return np.zeros((1, 21, len(coords)))

    # Prepare the list to store hand landmarks
    processed_hands = []

    for idx, lm in enumerate(lm_list):
        hand_landmarks = []  # Stores keypoints for one hand
        is_left_hand = results.multi_handedness[idx].classification[0].label.lower() == "left"
        for keypoint in keypoints:
            point = []  # Stores coordinates for a single keypoint
            for coord in coords:
                value = getattr(lm.landmark[keypoint], coord)
                # If using image coordinates and the hand is left, mirror the x-axis
                if coord == "x" and is_left_hand and not world:
                    value = 1 - value
                point.append(value)
            hand_landmarks.append(point)
        processed_hands.append(hand_landmarks)
    return np.array(processed_hands)


