#!/bin/bash
#SBATCH --job-name=language_identification
#SBATCH --account=s24o01-42-users
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=identification_err.txt

container_path=/ceph/hpc/data/s24o01-42-users/containers
nemo_version=24.12
container=${container_path}/nemo_${nemo_version}.sif

data_dir=/ceph/hpc/data/s24o01-42-users/corpuses/data_to_clean

WORK_DIR=/ceph/hpc/data/s24o01-42-users/scripts/language_identification

input_path=$1
output_path=$2
text_field=$3

# run with singularity
srun \
  --cpu-bind=verbose \
  --output="logs/${input_path}_identification.txt" \
  singularity exec \
    -B $WORK_DIR/model:/model,$WORK_DIR/dask_config:/config,$data_dir:/data \
    $container \
    python identify_languages.py \
      --input_path=/data/$input_path \
      --output_path=/data/$output_path \
      --text_field=$text_field
