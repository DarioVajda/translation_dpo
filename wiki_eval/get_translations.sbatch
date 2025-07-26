#!/bin/bash
#SBATCH --job-name=wiki-translate
#SBATCH --partition=frida
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/translate_%j.out

echo "Starting translation job on $(hostname) at $(date)"

# (Optional) activate your Python env if needed:
# source /mnt/env/bin/activate

srun \
    --container-image=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh \
    --container-mounts=$HOME/translate_env:/mnt/env,/shared/workspace/povejmo/translation_optimization:/workspace,/shared/workspace/povejmo:/povejmo \
    --container-workdir=/workspace/wiki_eval \
    python3 get_translations.py

echo "Done at $(date)"