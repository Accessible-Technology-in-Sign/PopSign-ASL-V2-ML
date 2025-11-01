#!/bin/bash

#SBATCH --job-name=Mediapipe_Holistic_563
#SBATCH --array=1-114
#SBATCH --account=
#SBATCH --partition=cpu-large
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --qos=inferno
#SBATCH --output=
#SBATCH --mail-type=NONE



module load anaconda3

conda activate pose-pipeline

INPUT_DIR="/path/to/input"
OUTPUT_DIR="/path/to/output"
MAX_JOBS=13

# Execute this under pose-pipeline/PACE-scripts
cd $SLURM_SUBMIT_DIR


# Read the workload for this job from the pre-generated file
job_line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs_signs.txt)
read -r -a signs <<< "$job_line"

cd ..

# Function to track running jobs and throttle if needed
function throttle_jobs {
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        sleep 0.5
    done
}
declare -a splits=("test" "train" "validation")

for split in "${splits[@]}"; do
    for sign in "${signs[@]}"; do
        sign_name=$(basename "$sign")
        input_sign_path="$INPUT_DIR/$split/$sign_name"
        out_sign_path="$OUTPUT_DIR/$split/$sign_name"
        mkdir -p "$out_sign_path"

        throttle_jobs  # Wait if too many jobs are already running

        # Launch the job in the background
        python3 process_sign.py holistic "$input_sign_path" "$out_sign_path" --writer h5py_metadata 2>&1 &
    done
done


# Wait for any remaining background jobs
wait



