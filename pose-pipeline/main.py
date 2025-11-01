import numpy as np
import os
import argparse
import importlib

def main():
    parser = argparse.ArgumentParser(description="Process video with a specified mediapipe solution.")

    parser.add_argument('solution', type=str, help="The solution to run.")
    parser.add_argument('input_file', type=str, help="Path to the input file.")
    parser.add_argument('output_file', type=str, help="Path to the output file.")
    
    parser.add_argument('--loader', type=str, default='cv2_loader', help="The name of the loader module (default: cv2_loader).")
    parser.add_argument('--writer', type=str, default='h5py_writer', help="The name of the writer module (default: h5py_writer).")

    args = parser.parse_args()
    
    loader_module = importlib.import_module(args.loader)
    load_file = getattr(loader_module, 'load_file')
    file_iters = getattr(loader_module, 'file_iters')
    
    solution_module = importlib.import_module(args.solution)
    run = getattr(solution_module, 'run')
    post_process = getattr(solution_module, 'post_process')

    writer_module = importlib.import_module(args.writer)
    write_file = getattr(writer_module, 'write_file')

    results = np.zeros((file_iters(os.path.join("/in", args.input_file)), 1, 21, 3)) #TODO, make rest dynamic
    for idx, frame in enumerate(load_file(os.path.join("/in", args.input_file))):
        results[idx, :, :, :] = post_process(run(frame))
    write_file(os.path.join("/out", args.output_file), results)
    

if __name__ == "__main__":
    import sys
    __file__ = "mputil"
    sys.argv[0] = "mputil"
    main()

