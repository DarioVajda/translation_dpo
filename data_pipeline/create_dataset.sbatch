#!/bin/bash
#SBATCH --job-name=translate
#SBATCH --output=log_create_dataset.txt
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:A100_80GB:3
#SBATCH --mem=64GB
#SBATCH --partition=frida

source_model_1=$1
source_model_2=$2

load_data_script=/ceph/hpc/data/s24o01-42-users/translation_optimization/get_translations/load_data_scripts/$3
output_folder=$4
data_id=$5

mkdir -p /shared/workspace/povejmo/translation_optimization/data_pipeline/$output_folder

# ---------------------------------------------------------------
# Getting the translations of the data using our two models
# ---------------------------------------------------------------
translations_folder=/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/$output_folder/translations
mkdir -p /shared/workspace/povejmo/translation_optimization/data_pipeline/$output_folder/translations

echo "Running translations for model 1: $source_model_1"
bash ../get_translations/run_translation_generic.sbatch \
    $source_model_1 \
    $load_data_script \
    $translations_folder/translation_1.jsonl \
    0.9 \
    $data_id
echo "Finished translations for model 1"

echo "Running translations for model 2: $source_model_2"
bash ../get_translations/run_translation_generic.sbatch \
    $source_model_2 \
    $load_data_script \
    $translations_folder/translation_2.jsonl \
    0.9 \
    $data_id
echo "Finished translations for model 2"


# ---------------------------------------------------------------
# Comet Scores
# ---------------------------------------------------------------
comet_scores_folder=/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/$output_folder/translations_with_scores
mkdir -p /shared/workspace/povejmo/translation_optimization/data_pipeline/$output_folder/translations_with_scores

bash ../comet_score/run_scoring_generic.sbatch \
    $translations_folder/translation_1_${data_id}.jsonl \
    $comet_scores_folder/translation_1_${data_id}.jsonl \
    8

bash ../comet_score/run_scoring_generic.sbatch \
    $translations_folder/translation_2_${data_id}.jsonl \
    $comet_scores_folder/translation_2_${data_id}.jsonl \
    8


# ---------------------------------------------------------------
# Language identification
# ---------------------------------------------------------------
language_id_folder=/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/$output_folder/language_id
mkdir -p /shared/workspace/povejmo/translation_optimization/data_pipeline/$output_folder/language_id

bash ../language_identification/identify_generic.bash \
    $comet_scores_folder/translation_1_${data_id}.jsonl \
    $language_id_folder/language_id_1_id${data_id}

bash ../language_identification/identify_generic.bash \
    $comet_scores_folder/translation_2_${data_id}.jsonl \
    $language_id_folder/language_id_2_id${data_id}


# ---------------------------------------------------------------
# Pair Data Creation
# ---------------------------------------------------------------
paired_data_folder=/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/$output_folder/paired_data
mkdir -p /shared/workspace/povejmo/translation_optimization/data_pipeline/$output_folder/paired_data

srun \
    --cpu-bind=verbose \
    --container-image /shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh \
    --container-mounts /shared/workspace/povejmo:/ceph/hpc/data/s24o01-42-users \
    --container-workdir /ceph/hpc/data/s24o01-42-users/translation_optimization/language_identification \
        bash -lc "python3 ../language_identification/pair_data_generic.py \
            --language_id_path1 $language_id_folder/language_id_1_id${data_id} \
            --language_id_path2 $language_id_folder/language_id_2_id${data_id} \
            --paired_data_path $paired_data_folder/paired_data_id${data_id}.jsonl"

# ---------------------------------------------------------------
# Creating the preference dataset
# ---------------------------------------------------------------
preference_data_folder=/ceph/hpc/data/s24o01-42-users/translation_optimization/data_pipeline/$output_folder/preference_data
mkdir -p /shared/workspace/povejmo/translation_optimization/data_pipeline/$output_folder/preference_data

# bad format examples:
srun \
    --cpu-bind=verbose \
    --container-image /shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh \
    --container-mounts /shared/workspace/povejmo:/ceph/hpc/data/s24o01-42-users \
    --container-workdir /ceph/hpc/data/s24o01-42-users/translation_optimization/language_identification \
        bash -lc "python3 ../preference_data/generic_scripts/bad_format_examples.py \
            --paired_data_path $paired_data_folder/paired_data_id${data_id}.jsonl \
            --output_path $preference_data_folder/bad_format_examples_id${data_id}.jsonl"

# bad lang examples:
srun \
    --cpu-bind=verbose \
    --container-image /shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh \
    --container-mounts /shared/workspace/povejmo:/ceph/hpc/data/s24o01-42-users \
    --container-workdir /ceph/hpc/data/s24o01-42-users/translation_optimization/language_identification \
        bash -lc "python3 ../preference_data/generic_scripts/bad_lang_examples.py \
            --paired_data_path $paired_data_folder/paired_data_id${data_id}.jsonl \
            --output_path $preference_data_folder/bad_lang_examples_id${data_id}.jsonl"

# choose examples (based on comet scores):
srun \
    --cpu-bind=verbose \
    --container-image /shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh \
    --container-mounts /shared/workspace/povejmo:/ceph/hpc/data/s24o01-42-users \
    --container-workdir /ceph/hpc/data/s24o01-42-users/translation_optimization/language_identification \
        bash -lc "python3 ../preference_data/generic_scripts/choose_examples.py \
            --paired_data_path $paired_data_folder/paired_data_id${data_id}.jsonl \
            --output_path $preference_data_folder/choose_examples_id${data_id}.jsonl"

# short examples (based on incorrect truncations):
srun \
    --cpu-bind=verbose \
    --container-image /shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh \
    --container-mounts /shared/workspace/povejmo:/ceph/hpc/data/s24o01-42-users \
    --container-workdir /ceph/hpc/data/s24o01-42-users/translation_optimization/language_identification \
        bash -lc "python3 ../preference_data/generic_scripts/short_examples.py \
            --paired_data_path $paired_data_folder/paired_data_id${data_id}.jsonl \
            --output_path $preference_data_folder/short_examples_id${data_id}.jsonl"

