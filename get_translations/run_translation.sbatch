#!/bin/bash
#SBATCH --job-name=translate
#SBATCH --account=s24o01-42-users
#SBATCH --output=translation_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:30:00
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --partition=gpu

model=$1
input_data=$2
output_path=$3
gpu_mem_util=$4

WORK_DIR=/ceph/hpc/data/s24o01-42-users

tp_size=${SLURM_GPUS_ON_NODE}

CONTAINER_DIR=$WORK_DIR/containers
VLLM_VERSION=v0.6.3.post1
CONTAINER=$CONTAINER_DIR/vllm-openai_${VLLM_VERSION}.sif

data_dir=$WORK_DIR/corpuses/wikipedia

export TOKENIZERS_PARALLELISM=false

export HF_TOKEN=$(<$HOME/hf_token.txt)

srun \
    --cpu-bind=verbose \
    --output="logs/${input_data}_${model}_translation_test.txt" \
    singularity exec \
        -B $data_dir:/data \
        $CONTAINER \
        python3 translate.py \
            --model_path=$model \
            --input_path=/data/$input_data \
            --output_path=/data/$output_path \
            --gpu_memory_util=$gpu_mem_util \
            --tp_size=$tp_size
