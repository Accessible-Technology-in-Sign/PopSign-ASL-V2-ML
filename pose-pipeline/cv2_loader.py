import cv2

def file_iters(filepath):
    cap  = cv2.VideoCapture(filepath)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

def load_file(filepath):
    cap  = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise IOError(f"Could not open Video File at {filepath}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def get_fps(filepath):
    # Open the fps and get the framerate
    cap  = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise IOError(f"Could not open Video File at {filepath}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps