#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=log_perform_eval.txt
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:H100:4
#SBATCH --mem=64GB
#SBATCH --partition=frida
#SBATCH --cpus-per-task=16

model=$1
load_eval_dataset=$2
output_path=$3

container_workdir=/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline
script_workdir=/shared/workspace/povejmo/translation_optimization/data_pipeline

echo "Model: $model"
echo "Load eval dataset script: $load_eval_dataset"
echo "Output path: $output_path"

# Create output directory if it does not exist
mkdir -p $output_path

# ---------------------------------------------------------------
# Generating the translations from the evaluation dataset
# ---------------------------------------------------------------
# bash ../get_translations/run_translation_generic.sbatch \
#     $model \
#     $container_workdir/$load_eval_dataset \
#     $container_workdir/$output_path/all_translation.jsonl \
#     0.9 \
#     0

# ---------------------------------------------------------------
# Language identification
# ---------------------------------------------------------------
language_id_path=$container_workdir/$output_path/language_id
bash ../language_identification/identify_generic.bash \
    $container_workdir/$output_path/all_translation_0.jsonl \
    $language_id_path

# ---------------------------------------------------------------
# Merging all translations with language labels in one file
# ---------------------------------------------------------------
merged_language_id_file=$language_id_path.jsonl
srun \
    --cpu-bind=verbose \
    --container-image /shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh \
    --container-mounts /shared/workspace/povejmo:/ceph/hpc/data/s24o01-42-users \
    --container-workdir /ceph/hpc/data/s24o01-42-users/translation_optimization/language_identification \
        bash -lc "python3 ../wiki_eval/merge_languages_generic.py \
                    --multilang_dir $language_id_path"

# ---------------------------------------------------------------
# Count bad language translations
# ---------------------------------------------------------------
# Make $output_path/results directory if it does not exist
mkdir -p $output_path/results
srun \
    --output=$output_path/results/count_bad_lang.txt \
    --cpu-bind=verbose \
    --container-image /shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh \
    --container-mounts /shared/workspace/povejmo:/ceph/hpc/data/s24o01-42-users \
    --container-workdir /ceph/hpc/data/s24o01-42-users/translation_optimization/language_identification \
        bash -lc "python3 ../wiki_eval/count_bad_lang_generic.py \
                    --multilang_file $merged_language_id_file"

# ---------------------------------------------------------------
# Count invalid truncation translations
# ---------------------------------------------------------------
mkdir -p $output_path/results
srun \
    --output=$output_path/results/count_short.txt \
    --cpu-bind=verbose \
    --container-image /shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh \
    --container-mounts /shared/workspace/povejmo:/ceph/hpc/data/s24o01-42-users \
    --container-workdir /ceph/hpc/data/s24o01-42-users/translation_optimization/language_identification \
        bash -lc "python3 ../wiki_eval/count_short_generic.py \
                    --sl_translations_file $merged_language_id_file"

# ---------------------------------------------------------------
# Scoring the slovene translations with comet
# ---------------------------------------------------------------
bash ../comet_score/run_scoring_generic.sbatch \
    $language_id_path/SL/all_translation_0.jsonl \
    $container_workdir/$output_path/scored_translations.jsonl \
    8 \
    $output_path/results/comet_scoring.txt

# Print the comet scoring results
srun \
    --cpu-bind=verbose \
    --container-image /shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh \
    --container-mounts /shared/workspace/povejmo:/ceph/hpc/data/s24o01-42-users \
    --container-workdir /ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline \
        bash -lc "python3 print_comet_scores.py \
                    --input_path $container_workdir/$output_path/scored_translations.jsonl \
                    --output_path $output_path/results/comet_scoring.txt"


# ---------------------------------------------------------------
# Judging the markdown format of the nemotron dataset
# ---------------------------------------------------------------
gemma_model_path=/ceph/hpc/data/s24o01-42-users/models/hf_models/gemma-3-27b-it

OUT_LOG_FILE=$(scontrol show job $SLURM_JOB_ID | awk -F= '/StdOut=/{print $2}')
sbatch --output=$OUT_LOG_FILE ../sft_translator/eval_md.sbatch \
    $container_workdir/$output_path/language_id.jsonl \
    $gemma_model_path \
    $container_workdir/$output_path/judged_translations.jsonl \
    $container_workdir/$output_path/results/markdown_judging.txt

# gemma_model_path=/shared/workspace/povejmo/models/hf_models/gemma-3-27b-it
echo "Finished performing evaluation for model $model"


# sbatch perform_eval.sbatch /ceph/hpc/data/s24o01-42-users/model_transfer/models/GaMS-9B-SFT-Translator-DPO load_all_eval_datasets.py gams_9b_sft_translator_dpo_eval