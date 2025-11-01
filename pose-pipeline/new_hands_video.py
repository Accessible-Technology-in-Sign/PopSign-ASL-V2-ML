import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# This uses the new version of mediapipe (0.10.21), for which the documentation 
# can be found here: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python#image_2
#
# You can download the mediapipe task by executing the command below (provided in the example documentation)
# wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Mediapipe hand landmark options
# Documentation for the options can be found here: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python#image_1
#   - By default, num_hands is 1, while every other value is 0.5.
#   - The model path is necessary to run the processing and it has to be absolute
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='/path/to/PopSignV2-ML/pose-pipeline/mediapipe_models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)

def run(frame, frame_timestamp_ms, colorspace_convert=cv2.COLOR_BGR2RGB, **kwargs):
    if 'flipH' in kwargs and kwargs['flipH']:
        cv2.flip(frame, 1)
    if 'flipV' in kwargs and kwargs['flipV']:
        cv2.flip(frame, 0)

    if colorspace_convert:
        frame = cv2.cvtColor(frame, colorspace_convert)


    with HandLandmarker.create_from_options(options) as landmarker:
        # Convert to a mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    #return __hands.process(frame)

    return hand_landmarker_result


def post_process(results, world=False, coords=["x", "y", "z"]):
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

    lm_list = None
    if world: 
        lm_list = results.hand_world_landmarks
    else:
        lm_list = results.hand_landmarks

    # If no hands are detected, return a zero array with the expected shape
    if not lm_list:
        return np.zeros((1, 21, len(coords))), {"success": False, "left_hand": False}


    # Landmarks are stored in a shape of:
    #[num_hands, landmarks, xyz]
    hand_landmarks = lm_list[0] # Since there is only one hand, take the first landmark
    landmark_array = []
    is_left_hand = results.handedness[0][0].display_name.lower() == "left" # Metadata to store for augmentation later 

    # Since there is only 21 hand landmarks, iterate over all the landmarks
    for i in range(21):
        point = []
        # coord is one of "x", "y", "z"
        for coord in coords:
            value = getattr(hand_landmarks[i], coord)
            point.append(value)
        landmark_array.append(point)
    
    landmark_array = np.array(landmark_array)
    landmark_array = landmark_array[np.newaxis, :] # Add a new axis here to be consistent with previous data formats
    return landmark_array, {"success": True, "left_hand": is_left_hand}



