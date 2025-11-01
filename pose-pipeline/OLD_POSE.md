# Pose Pipeline

The Pose Pipeline is responsible for converting all videos from `*.mp4` format to a pose format ready for
the machine learning pipeline. 

## Usage:

To build the Docker container to run this script do:

```
docker build -t container-name .
```

Once the container has been built, to process a specific file, run the following command:

```
docker run container-name [solution] [input-file] [output-file] --loader /path/to/loader --writer /path/to/writer
```

### PACE-specific instructions:

Georgia Tech's super-cluster, PACE, uses *apptainer* to manage containers. We can use apptainer to
run docker containers on PACE (which allows to import our specific libaries in our programs). In order
to upload the container to PACE, the following steps must be taken:

1. On a local machine, build the container using docker.
1. Login to Docker and push it to DockerHub
    ```
    docker login -u USERNAME
    docker push container-name
    ```
1. On PACE, download the container that was pushed to Dockerhub.
    > This can only be done in either an `*.sbatch` script or an OnDemand environment. This command cannot be run in an ssh tunnel because all (significant) compute on PACE needs to attached to a specific payment account. 
    ```
    apptainer pull container.sif docker://username/container-name
    ```
1. Run an `*.sbatch` script to process all files within a dataset. Refer to `PACE-scripts/run_mediapipe.sbatch` for more details.

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


