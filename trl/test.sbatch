#!/bin/bash
#SBATCH --job-name=translator_test
#SBATCH --account=s24o01-42-users
#SBATCH --output=test_output.txt
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive

#CONTAINER=/ceph/hpc/data/s24o01-42-users/containers/transformers_deepspeed_latest.sif
CONTAINER=/ceph/hpc/data/s24o01-42-users/containers/slobench_vllm.sif

# Execute the command
srun \
    --cpu-bind=verbose \
    singularity exec \
        $CONTAINER \
        python3 test_model.py

