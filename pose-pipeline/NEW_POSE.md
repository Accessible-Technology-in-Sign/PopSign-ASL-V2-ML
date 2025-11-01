# Pose Pipeline

The Pose Pipeline is responsible for converting all videos from `*.mp4` format to a pose format ready for
the machine learning pipeline. 

## Usage:

```
python3 process_sign.py <SOLUTION> <INPUT_SIGN_PATH> <OUTPUT_SIGN_PATH> --writer <WRITER> --LOADER <LOADER>
```

### PACE-specific instructions:


1. Run an `*.sbatch` script to process all files within a dataset. Refer to `PACE-scripts/process_mediapipe_holistic.sbatch` for more details.

## Details

### Solution 

Solution refers to which pose / mediapipe library you would like to use when processing the videos. 
The default solution is `hands.py` which currently does single-hand pose processing. 
If you would like to use a different pose solution, use `hands.py` as a template, and ensure that it
implements both `run()` and `post_process()`. Once it has been created, specify the base name without `*.py` in the command line argument


### Loader

Loader refers to the library that is used to load in the videos. The default library used is CV2 as 
implemented in `cv2_loader.py`. If you would like to use a different video loading library, use 
`cv2_loader.py` as a template, and ensure that it implements both `file_iters()` and `load_file()`. Once it has been created, include the following argument when processing the videos:

```
--loader /path/to/loader.py
```

### Writer
Writer refers to the output format of data that is used in the pose dataset. The default file format used
is `*.h5` as implemented in `h5py_writer.py`. If you would like to use a different file format, use
`h5py_writer.py` as a template, and ensure that it implements `write_file()`. Once it has been created,
include the following argument when processing the videos: 

```
--writer /path/to/writer.py
```


