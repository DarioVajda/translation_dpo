#!/bin/bash
#SBATCH --job-name=translate
#SBATCH --output=log_perform_eval.txt
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64GB
#SBATCH --partition=frida

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
bash ../get_translations/run_translation_generic.sbatch \
    $model \
    $container_workdir/$load_eval_dataset \
    $container_workdir/$output_path/all_translation.jsonl \
    0.9 \
    0

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
    $comet_scores_folder/translation_1_${data_id}.jsonl \
    8 \
    $output_path/results/comet_scoring.txt


echo "Finished performing evaluation for model $model"
