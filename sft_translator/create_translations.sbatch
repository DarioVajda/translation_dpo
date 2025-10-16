#!/bin/bash
#SBATCH --job-name=create_dataset
#SBATCH --output=logs/create_dataset_%j.out
#SBATCH --partition=nxt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:GH200:1
#SBATCH --time=8:00:00
#SBATCH --mem=128G

output_folder=$1
mkdir -p /shared/workspace/povejmo/translation_optimization/get_translations/$output_folder

data_id=$2

translate_script=../get_translations/run_translation_generic_gh.sbatch
# translate_script=../get_translations/run_translation_generic.sbatch


load_data_script=/ceph/hpc/data/s24o01-42-users/translation_optimization/get_translations/load_data_scripts/load_nemotron.py

bash $translate_script \
    GaMS-Beta/GaMS-9B-SFT-Translator \
    $load_data_script \
    $output_folder/translations_${data_id}.jsonl \
    0.95 \
    $data_id

