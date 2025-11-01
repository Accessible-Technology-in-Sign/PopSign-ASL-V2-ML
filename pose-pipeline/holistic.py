import mediapipe as mp
import cv2
import os
import numpy as np

# This uses a legacy version of a mediapipe solution
# The documentation for the legacy version can be found here: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
# Support for it has ended on March 1st, 2023: https://ai.google.dev/edge/mediapipe/solutions/guide#legacy 

mp_holistic = mp.solutions.holistic
__holistic = mp_holistic.Holistic(
    static_image_mode=bool(os.environ.get("STATIC_IMAGE_MODE", False)),
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=float(os.environ.get("MIN_DETECTION_CONFIDENCE", 0.5)),
    min_tracking_confidence=float(os.environ.get("MIN_TRACKING_CONFIDENCE", 0.5)))

def run(frame, colorspace_convert=cv2.COLOR_BGR2RGB, **kwargs):
    if 'flipH' in kwargs and kwargs['flipH']:
        cv2.flip(frame, 1)
    if 'flipV' in kwargs and kwargs['flipV']:
        cv2.flip(frame, 0)

    if colorspace_convert:
        frame = cv2.cvtColor(frame, colorspace_convert)
    return __holistic.process(frame)

def post_process(results, coords="xyz", world=False):
    """
    Processes landmarks from MediaPipe Holistic results.

    Parameters:
    - results: The output from MediaPipe Holistic processing.
    - coords (str): Specifies which coordinate dimensions to extract (e.g., "xyz" or "xy").
    - world (bool): If True, use world coordinates (3D). If False, use image coordinates (2D).
    - mirror_left_hand (bool): If True and using image coordinates, mirror the x-axis for the left hand.

    Returns:
    - A dictionary containing NumPy arrays for 'face', 'pose', 'left_hand', and 'right_hand' landmarks.
      Each array has the shape (num_landmarks, len(coords)). If a component is not detected, its value is None.
    """
    def extract_landmarks(landmarks, num_landmarks):
        if landmarks:
            landmark_array = []
            for idx in range(num_landmarks):
                point = []
                for coord in coords:
                    value = getattr(landmarks.landmark[idx], coord)
                    point.append(value)
                landmark_array.append(point)
            return np.array(landmark_array), True
        else:
            # return None
            return np.zeros((num_landmarks, len(coords))), False
        

    if world:
        face_landmarks, has_face = extract_landmarks(results.face_landmarks, 468)
        pose_landmarks, has_pose = extract_landmarks(results.pose_world_landmarks, 33)
        left_hand_landmarks, has_left = extract_landmarks(results.left_hand_world_landmarks, 21)
        right_hand_landmarks, has_right = extract_landmarks(results.right_hand_world_landmarks, 21)
        
        output_array = np.concatenate((face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks))
        
        metadata = {"has_face": has_face, "has_pose": has_pose, "has_left": has_left, "has_right": has_right}
        return output_array, metadata
    else:
        face_landmarks, has_face = extract_landmarks(results.face_landmarks, 468)
        pose_landmarks, has_pose = extract_landmarks(results.pose_landmarks, 33)
        left_hand_landmarks, has_left = extract_landmarks(results.left_hand_landmarks, 21)
        right_hand_landmarks, has_right = extract_landmarks(results.right_hand_landmarks, 21)
        
        #print(f"face_landmarks shape: {face_landmarks.shape}")
        #print(f"pose_landmarks shape: {pose_landmarks.shape}")
        #print(f"left shape: {left_hand_landmarks.shape}")
        #print(f"right shape: {right_hand_landmarks.shape}")
        output_array = np.concatenate((face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks))
        
        metadata = {"has_face": has_face, "has_pose": has_pose, "has_left": has_left, "has_right": has_right}
        return output_array, metadata



