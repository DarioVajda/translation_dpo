#!/bin/bash
#SBATCH --job-name=translate
#SBATCH --account=s24o01-42-users
#SBATCH --output=translation_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --partition=gpu

model=$1
input_data=$2
output_path=$3
gpu_mem_util=$4
id=$5
dataset_name=$6

if [ "$dataset_name" = "wiki" ]; then
  TRANSLATE_FILE="translate_wiki.py"
elif [ "$dataset_name" = "ccnews" ]; then
  TRANSLATE_FILE="translate_ccnews.py"
else
  echo "Error: unknown dataset_name '$dataset_name'" >&2
  exit 1
fi

WORK_DIR=/ceph/hpc/data/s24o01-42-users

tp_size=${SLURM_GPUS_ON_NODE}

CONTAINER_DIR=$WORK_DIR/containers
# CONTAINER=$CONTAINER_DIR/slobench_vllm.sif
CONTAINER=$CONTAINER_DIR/vllm-openai_v0.6.3.post1.sif

MODELS_DIR=/ceph/hpc/data/s24o01-42-users/models/hf_models

data_dir=$WORK_DIR/corpuses/wikipedia

export TOKENIZERS_PARALLELISM=false

export HF_TOKEN=$(<$HOME/hf_token.txt)
export HF_HOME=

export VLLM_DISABLE_COMPILE_CACHE=1

srun \
    --cpu-bind=verbose \
    --output="new_logs/${model}_translation_${dataset_name}_${id}.txt" \
    singularity exec \
        -B $data_dir:/data,$MODELS_DIR:/models \
        $CONTAINER \
        python3 ${python_file} \
            --model_path=$model \
            --input_path=/data/$input_data \
            --output_path=/data/$output_path \
            --gpu_memory_util=$gpu_mem_util \
            --tp_size=$tp_size \
            --id=$id 
