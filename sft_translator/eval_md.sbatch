#!/bin/bash
#SBATCH --job-name=judge_markdown
#SBATCH --output=judging2_log.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:GH200:1
#SBATCH --mem=128GB
#SBATCH --partition=nxt

# -----------------------------------------------------------------------------------------------------
# Navigate to the sft_translator directory
# -----------------------------------------------------------------------------------------------------
current_path=$(pwd)
cd /shared/workspace/povejmo/translation_optimization/sft_translator
# -----------------------------------------------------------------------------------------------------

# bash ../sft_translator/eval_md.sbatch     /ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/gams_9b_instruct_dpo_eval/language_id.jsonl     /ceph/hpc/data/s24o01-42-users/models/hf_models/gemma-3-27b-it     /ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/gams_9b_instruct_dpo_eval/judged_translations
input_path=$1
model=$2
output_path=$3
results_path=$4

echo "Starting judging with the following parameters:"
echo "Input path: $input_path"
echo "Judge model: $model"
echo "Output path: $output_path"
echo "Results path: $results_path"

JUDGING_FILE=/ceph/hpc/data/s24o01-42-users/translation_optimization/sft_translator/judge_markdown_generic.py

# WORK_DIR=/ceph/hpc/data/s24o01-42-users
WORK_DIR=/shared/workspace/povejmo

tp_size=${SLURM_GPUS_ON_NODE}

CONTAINER_DIR=$WORK_DIR/containers
# CONTAINER=$CONTAINER_DIR/slobench_vllm_latest.sqsh
CONTAINER=$CONTAINER_DIR/vllm_gh_cuda_12.4.1.sqsh

export TOKENIZERS_PARALLELISM=false

export VLLM_DISABLE_COMPILE_CACHE=1

# Create output directory if it doesn't exist
# mkdir $output_path


srun \
    --cpu-bind=verbose \
    --container-image $CONTAINER \
    --container-mounts /shared/workspace/povejmo:/ceph/hpc/data/s24o01-42-users \
    --container-workdir /ceph/hpc/data/s24o01-42-users/translation_optimization/sft_translator \
        bash -c "
            echo 'starting...' && \
            pip install datasets && \
            python3 ${JUDGING_FILE} \
                --input_path=$input_path \
                --model_path=$model \
                --results_path=$results_path \
                --output_path=$output_path \
                --gpu_memory_util=0.95 \
                --tp_size=$tp_size \
                --text_field=sl_translation"


# -----------------------------------------------------------------------------------------------------
# Go back to the original path
# -----------------------------------------------------------------------------------------------------
cd $current_path
# -----------------------------------------------------------------------------------------------------