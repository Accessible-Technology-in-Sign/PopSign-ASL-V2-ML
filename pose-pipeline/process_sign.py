import numpy as np
import os
import argparse
import importlib
#from tqdm import tqdm
#from p_tqdm import p_map
#import multiprocessing
#multiprocessing.set_start_method("forkserver", force=True)

def main():
    parser = argparse.ArgumentParser(description="Process video with a specified mediapipe solution.")

    parser.add_argument('solution', type=str, help="The solution to run.")
    parser.add_argument('input_sign_path', type=str, help="Path to the input sign")
    parser.add_argument('output_sign_path', type=str, help="Path to the output sign")
    
    parser.add_argument('--loader', type=str, default='cv2_loader', help="The name of the loader module (default: cv2_loader).")
    parser.add_argument('--writer', type=str, default='h5py_writer', help="The name of the writer module (default: h5py_writer).")

    #input_dir = "/data/sign_language_videos/popsign_v2/563/videos/popsign_v2_combined"
    #output_dir = "/data/sign_language_videos/popsign_v2/563/mediapipe/normalized"
    
    args = parser.parse_args()

    input_dir = args.input_sign_path
    output_dir = args.output_sign_path
    
    loader_name = args.loader
    loader_module = importlib.import_module(args.loader)
    load_file = getattr(loader_module, 'load_file')
    file_iters = getattr(loader_module, 'file_iters')
    get_fps = getattr(loader_module, 'get_fps')
    
    solution_name = args.solution
    solution_module = importlib.import_module(args.solution)
    run = getattr(solution_module, 'run')
    post_process = getattr(solution_module, 'post_process')

    writer_name = args.writer
    writer_module = importlib.import_module(args.writer)
    write_file = getattr(writer_module, 'write_file')


    video_files = os.listdir(input_dir)
    for video_file in video_files:
        input_file = os.path.join(input_dir, video_file)
        #file_output_dir = os.path.join(output_dir, split, sign)
        file_output_dir = output_dir 

        try:

            video_fps = get_fps(input_file)
            results = np.zeros((file_iters(input_file), 1, 21, 3))
            if solution_name == "new_hands" or solution_name == "new_hands_video":
                metadata = {
                    "video_height": 0,
                    "video_width": 0,
                    "available_frames": 0,
                    "missing_frames": 0,
                    "total_frames": 0,
                    "left_hand": False
                }
            elif solution_name == "holistic":
                metadata = {
                    "video_height": 0,
                    "video_width": 0,
                    "has_face": 0,
                    "has_pose": 0,
                    "has_left": 0,
                    "has_right": 0,
                    "available_frames": 0,
                    "missing_frames": 0,
                    "total_frames": 0,
                }
                results = np.zeros((file_iters(input_file), 1, 543, 3))
            
            for idx, frame in enumerate(load_file(input_file)):

                # If using the new_hands to generate mediapipe, need more code to handle the metadata
                if solution_name == "new_hands" or solution_name == "new_hands_video":

                    if solution_name == "new_hands_video":
                        # 1000 (ms / sec) * (sec / frames) * frame = ms
                        frame_timestamps_ms = int((1000 / video_fps) * idx) 
                        hand_landmarks = run(frame, frame_timestamps_ms)
                        print("Using new_hands_video")
                    else:
                        hand_landmarks = run(frame)
                    
                    data, frame_metadata = post_process(hand_landmarks, world=False)
                    if frame_metadata["success"]:
                        metadata["available_frames"] += 1
                    else:
                        metadata["missing_frames"] += 1

                    metadata["total_frames"] += 1
                    metadata["left_hand"] = metadata["left_hand"] or frame_metadata["left_hand"]  # Set left_hand to true if we have seen a left hand at one point

                    # Only take the first frame for metadata
                    if idx == 0:
                        metadata["video_height"] = frame.shape[0]
                        metadata["video_width"] = frame.shape[1]


                elif solution_name == "holistic":
                    all_landmarks = run(frame)
                    data, frame_metadata = post_process(all_landmarks, world=False)

                    if not (frame_metadata['has_face'] or frame_metadata['has_pose'] or frame_metadata['has_left'] or frame_metadata['has_right']):
                        metadata['available_frames'] += 1
                    else:
                        metadata["missing_frames"] += 1

                    if frame_metadata['has_face']:
                        metadata['has_face'] += 1
                    if frame_metadata['has_pose']:
                        metadata['has_pose'] += 1
                    if frame_metadata['has_left']:
                        metadata['has_left'] += 1
                    if frame_metadata['has_right']:
                        metadata['has_right'] += 1
                    
                    metadata['total_frames'] += 1

                    if idx == 0:
                        metadata["video_height"] = frame.shape[0]
                        metadata["video_width"] = frame.shape[1]

                else:
                    # If there is no metadata, then just write as is
                    data = post_process(run(frame), world=False)

                results[idx, :, :, :] = data


            basename = os.path.basename(input_file)
            output_path = os.path.join(file_output_dir, f"{basename}.h5")

            if writer_name == "h5py_metadata":
                write_file(output_path, results, metadata)
            else:
                write_file(output_path, results)

        except Exception as e:
            print(f"ERROR {input_file}: {e}")
    
    print(f"Finished {input_dir}")

    

if __name__ == "__main__":
    import sys
    __file__ = "mputil"
    sys.argv[0] = "mputil"
    main()

