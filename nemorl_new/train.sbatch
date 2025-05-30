#!/bin/bash
#SBATCH --job-name=dpo_fine_tuning
#SBATCH --output=dpo_test_run_output.txt
#SBATCH --error=dpo_test_run_error.txt
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --exclusive

SCRIPT_DIR="/ceph/hpc/data/s24o01-42-users/translation_optimization/nemorl_new"

RESULTS_DIR="$SCRIPT_DIR/result_dir"
OUTFILE="${RESULTS_DIR}/rm-%j_%t.out"
CONTAINER=/ceph/hpc/data/s24o01-42-users/containers/nemo_rl_installed.sqsh
mkdir -p ${RESULTS_DIR}

DATA_DIR=/ceph/hpc/data/s24o01-42-users/translation_optimization/preference_data

export WANDB_API_KEY="79af9dbae290344f5c04c8069ac6475d3b231866"

srun \
    --cpu-bind=verbose \
    --output=$OUTFILE \
    singularity exec \
        --nv \
        -B $DATA_DIR:/data,$SCRIPT_DIR:/script,$RESULTS_DIR:/results \
        --pwd /opt/nemo-rl \
        $CONTAINER \
        /script/train_script.sh

echo "Training completed!"
echo "Outputs are saved in ${RESULTS_DIR}"
