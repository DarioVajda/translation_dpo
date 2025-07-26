#!/bin/bash
#SBATCH --job-name=translate
#SBATCH --output=translation_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=64GB
#SBATCH --partition=frida

model=$1
load_data_script=$2
output_path=$3
gpu_mem_util=$4
id=$5

TRANSLATE_FILE=/ceph/hpc/data/s24o01-42-users/translation_optimization/get_translations/translate_generic.py
echo "Translate file: $TRANSLATE_FILE"

# WORK_DIR=/ceph/hpc/data/s24o01-42-users
WORK_DIR=/shared/workspace/povejmo

tp_size=${SLURM_GPUS_ON_NODE}

CONTAINER_DIR=$WORK_DIR/containers
# CONTAINER=$CONTAINER_DIR/slobench_vllm.sif
# CONTAINER=$CONTAINER_DIR/vllm-openai_v0.6.3.post1.sif
CONTAINER=$CONTAINER_DIR/slobench_vllm_latest.sqsh

# MODELS_DIR=/ceph/hpc/data/s24o01-42-users/models/hf_models
# MODELS_DIR=/shared/workspace/povejmo/models/hf_model
MODELS_DIR=$WORK_DIR/translation_optimization/models

data_dir=$WORK_DIR/corpuses/wikipedia

export TOKENIZERS_PARALLELISM=false

export HF_TOKEN=$(<$HOME/hf_token.txt)
export HF_HOME=

export VLLM_DISABLE_COMPILE_CACHE=1

# srun \
#     --cpu-bind=verbose \
#     --output="new_logs/${model}_translation_${dataset_name}_${id}.txt" \
#     singularity exec \
#         -B $data_dir:/data,$MODELS_DIR:/models \
#         $CONTAINER \
#         python3 ${python_file} \
#             --model_path=$model \
#             --input_path=/data/$input_data \
#             --output_path=/data/$output_path \
#             --gpu_memory_util=$gpu_mem_util \
#             --tp_size=$tp_size \
#             --id=$id 
echo "python3 translate_generic.py \
              --model_path=$model \
              --input_module=$load_data_script \
              --output_path=$output_path \
              --gpu_memory_util=$gpu_mem_util \
              --tp_size=$tp_size \
              --id=$id"

srun \
    --cpu-bind=verbose \
    --container-image $CONTAINER \
    --container-mounts $data_dir:/data,$MODELS_DIR:/models,/shared/workspace/povejmo:/ceph/hpc/data/s24o01-42-users \
    --container-workdir /ceph/hpc/data/s24o01-42-users/translation_optimization/get_translations \
        bash -lc "pip install --upgrade vllm==0.9.2 && \
            python3 translate_generic.py \
              --model_path=$model \
              --input_module=$load_data_script \
              --output_path=$output_path \
              --gpu_memory_util=$gpu_mem_util \
              --tp_size=$tp_size \
              --id=$id"
