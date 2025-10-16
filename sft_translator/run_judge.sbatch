#!/bin/bash
#SBATCH --job-name=judge_markdown
#SBATCH --output=judging2_log.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:GH200:1
#SBATCH --mem=128GB
#SBATCH --partition=nxt

# sbatch run_judge.sbatch /shared/workspace/povejmo/models/hf_models/gemma-3-27b-it judged3_test
model=$1
output_path=$2

JUDGING_FILE=/shared/workspace/povejmo/translation_optimization/sft_translator/judge_markdown.py

# WORK_DIR=/ceph/hpc/data/s24o01-42-users
WORK_DIR=/shared/workspace/povejmo

tp_size=${SLURM_GPUS_ON_NODE}

CONTAINER_DIR=$WORK_DIR/containers
CONTAINER=$CONTAINER_DIR/vllm_gh_cuda_12.4.1.sqsh
# CONTAINER=/shared/workspace/povejmo/containers/slobench_vllm_amd_latest.sqsh

# MODELS_DIR=/ceph/hpc/data/s24o01-42-users/models/hf_models
MODELS_DIR=/shared/workspace/povejmo/models/hf_model
# MODELS_DIR=$WORK_DIR/translation_optimization/models

data_dir=$WORK_DIR/corpuses/wikipedia

export TOKENIZERS_PARALLELISM=false

# export HF_TOKEN=$(<$HOME/hf_token.txt)
# export HF_HOME=

export VLLM_DISABLE_COMPILE_CACHE=1


# Create output directory if it doesn't exist
mkdir $output_path

srun \
    --cpu-bind=verbose \
    --container-image $CONTAINER \
    --container-mounts /shared/workspace/povejmo:/shared/workspace/povejmo \
    --container-workdir /shared/workspace/povejmo/translation_optimization/sft_translator \
        bash -c "echo 'starting...' && pip install datasets && python3 ${JUDGING_FILE} \
            --model_path=$model \
            --output_path=$output_path \
            --gpu_memory_util=0.9 \
            --tp_size=$tp_size \
            --text_field=gams_27b_translation"

# srun \
#     --cpu-bind=verbose \
#     --container-image $CONTAINER \
#     --container-mounts /shared/workspace/povejmo:/shared/workspace/povejmo \
#     --container-workdir /shared/workspace/povejmo/translation_optimization/sft_translator \
#         bash -c "echo 'starting...' && pip install datasets && python3 ${JUDGING_FILE} \
#             --model_path=$model \
#             --output_path=$output_path \
#             --gpu_memory_util=0.9 \
#             --tp_size=$tp_size \
#             --text_field=sl_translation"
