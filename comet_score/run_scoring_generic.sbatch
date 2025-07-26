#!/bin/bash
#SBATCH --job-name=comet_scoring
#SBATCH --partition=frida
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=scoring_err.txt

# WORK_DIR=/ceph/hpc/data/s24o01-42-users
WORK_DIR=/shared/workspace/povejmo

container_dir=$WORK_DIR/containers
nemo_version=24.12
# container=${container_dir}/nemo_${nemo_version}.sif
container=/shared/workspace/povejmo/containers/nemo_24.12.sqsh

data_dir=$WORK_DIR/data

export HF_TOKEN=$(<$HOME/hf_token.txt)
export HF_HOME=$WORK_DIR/huggingface_cache

input_path=$1
output_path=$2
batch_size=$3
log_path=$4

# sbatch run_scoring.sbatch /ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/intersection/sft/eurollm_translations_filtered.jsonl /ceph/hpc/data/s24o01-42-users/translation_optimization/wiki_eval/comet_scores/sft/eurollm_translations.jsonl 8

echo "Starting COMET scoring"

# run with singularity
srun \
  --output=$log_path \
  --cpu-bind=verbose \
  --container-image $container \
  --container-mounts $data_dir:/data,/shared/workspace/povejmo:/ceph/hpc/data/s24o01-42-users \
  --container-workdir /ceph/hpc/data/s24o01-42-users/translation_optimization/comet_score \
    ./install_and_run.sh $input_path $output_path $batch_size
  # singularity exec \
  #   -B $data_dir:/data \
  #   $container \
  #   ./install_and_run.sh $input_path $output_path $batch_size

echo "COMET scoring done"