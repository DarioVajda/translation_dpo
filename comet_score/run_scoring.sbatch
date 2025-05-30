#!/bin/bash
#SBATCH --job-name=comet_scoring
#SBATCH --account=s24o01-42-users
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --output=scoring_err.txt

WORK_DIR=/ceph/hpc/data/s24o01-42-users

container_dir=$WORK_DIR/containers
nemo_version=24.12
container=${container_dir}/nemo_${nemo_version}.sif

data_dir=$WORK_DIR/data

export HF_TOKEN=$(<$HOME/hf_token.txt)
export HF_HOME=$WORK_DIR/huggingface_cache

input_path=$1
output_path=$2
batch_size=$3

# run with singularity
srun \
  --cpu-bind=verbose \
  --output="logs/${input_path}_scoring.txt" \
  singularity exec \
    -B $data_dir:/data \
    $container \
    ./install_and_run.sh $input_path $output_path $batch_size