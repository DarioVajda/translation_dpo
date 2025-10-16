#!/bin/bash
#SBATCH --job-name=dpo_fine_tuning
#SBATCH --output=full_parameter_training.txt
#SBATCH --time=7-00
#SBATCH --partition=frida
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:B200:8
#SBATCH --mem=0
#SBATCH --reservation=bench_ixb
#SBATCH --exclusive

# Get the list of hostnames from SLURM
mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

HOSTFILE_PATH="/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/result_dir/hostfile_${SLURM_JOB_ID}"
echo "Creating hostfile at ${HOSTFILE_PATH}"
: > "${HOSTFILE_PATH}" # Clear/create
mapfile -t SLURM_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -u)
for NODE_NAME in "${SLURM_NODES[@]}"; do
    echo "${NODE_NAME} slots=${SLURM_GPUS_ON_NODE}" >> "${HOSTFILE_PATH}"
done
cat "${HOSTFILE_PATH}"

grep ^tty /etc/group


NUM_NODES=$SLURM_JOB_NUM_NODES      # Total number of nodes you are using
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE   # Number of GPUs on each node
MASTER_ADDR=${HOSTS[0]}             # IP address (acrually, the name for which dns is configured) of the MASTER node (first node in the list)
MASTER_PORT=29500                   # A free port on the MASTER node for communication

# Calculate total number of processes/GPUs
TOTAL_PROCESSES=$(($NUM_NODES * $GPUS_PER_NODE))

# Container path
NEMO_VERSION=25.07
CONTAINER=/shared/workspace/povejmo/containers/nemo_${NEMO_VERSION}.sqsh  # <---- TODO: CHANGE TO B200 COMPATIBLE CONTAINER

# Path to your training script
TRAINING_SCRIPT="train.py"

# Get the parameters for the training script
DPO_RANK=${1:-0} # 0 -> FULL PARAMETER FINE-TUNING, >0 -> LoRA FINE-TUNING (with given rank)
LEARNING_RATE=${2:-1e-7}
BETA=${3:-0.1}
EPOCHS=${4:-4}
# MODEL=${5:-"cjvt/GaMS-9B-Instruct"}
MODEL=${5:-"GaMS-Beta/GaMS-9B-SFT-Translator"}

# SSL Certificate Configuration
#CACERT_FILE_PATH="/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/cacert.pem"
#export SSL_CERT_FILE="${CACERT_FILE_PATH}"
#export REQUESTS_CA_BUNDLE="${CACERT_FILE_PATH}"

# Training script arguments
SCRIPT_ARGS="--rank=${DPO_RANK} --learning_rate=${LEARNING_RATE} --total_epochs=${EPOCHS} --beta=${BETA} --model=${MODEL}"

# Accelerate config file path
ACCELERATE_CONFIG_FILE=/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/accelerate_config_b200.yaml

# Set NCCL debug to info for more detailed logging if issues persist
# export NCCL_DEBUG=INFO

# Export wandb key
export WANDB_API_KEY="79af9dbae290344f5c04c8069ac6475d3b231866"

# Preparing the results directory and output files
RESULTS_DIR="/shared/workspace/povejmo/translation_optimization/trl/logs_dir_b200"
OUTFILE="${RESULTS_DIR}/rm-%j_%t.out"
ERRFILE="${RESULTS_DIR}/rm-%j_%t.err"

mkdir -p ${RESULTS_DIR}

# Writing the launch command using accelerate in the cmd variable
read -r -d '' cmd <<EOF
pip3 install deepspeed && \
pip3 install trl && \
pip3 install tokenizers==0.22.0 && \
echo "*******STARTING********" \
&& export DS_DEBUG=1 \
&& echo "--- Running on Node Rank: \$SLURM_PROCID ---" \
&& echo "Total Nodes: $NUM_NODES" \
&& echo "--- CUDA_VISIBLE_DEVICES for this srun task: \$CUDA_VISIBLE_DEVICES ---" \
&& echo "GPUs per Node: $GPUS_PER_NODE" \
&& echo "Master Address: $MASTER_ADDR" \
&& echo "Master Port: $MASTER_PORT" \
&& echo "-------------------------------------------" \
&& echo "accelerate launch \
    --config_file "$ACCELERATE_CONFIG_FILE" \
    --num_machines "$NUM_NODES" \
    --machine_rank "\$SLURM_PROCID" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --num_processes "$TOTAL_PROCESSES" \
    --deepspeed_multinode_launcher "standard" \
    --rdzv_backend static \
    --same_network \
    --mixed_precision bf16 \
    \"$TRAINING_SCRIPT\" \
    $SCRIPT_ARGS" \
&& echo "-------------------------------------------" \
&& accelerate launch \
    --config_file "$ACCELERATE_CONFIG_FILE" \
    --num_machines "$NUM_NODES" \
    --machine_rank "\$SLURM_PROCID" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --num_processes "$TOTAL_PROCESSES" \
    --deepspeed_multinode_launcher "standard" \
    --rdzv_backend static \
    --same_network \
    --mixed_precision bf16 \
    "$TRAINING_SCRIPT" \
    $SCRIPT_ARGS \
&& echo "--- Script finished on Node Rank: \$SLURM_PROCID ---"
EOF

# Execute the command
# srun \
#     --cpu-bind=verbose \
#     --job-name=${SLURM_JOB_NAME} \
#     --nodes=${SLURM_JOB_NUM_NODES} \
#     --ntasks-per-node=1 \
#     --gpus-per-task=$GPUS_PER_NODE \
#     --output=$OUTFILE \
#     singularity exec \
#         --nv \
#         $CONTAINER \
#         bash -c "${cmd}"

srun \
    --cpu-bind=verbose \
    --job-name=${SLURM_JOB_NAME} \
    --output=$OUTFILE \
    --nodes=${SLURM_JOB_NUM_NODES} \
    --ntasks-per-node=1 \
    --gpus-per-task=$GPUS_PER_NODE \
    --container-image=$CONTAINER \
    --container-mounts /shared/workspace/povejmo:/ceph/hpc/data/s24o01-42-users \
    --container-workdir /ceph/hpc/data/s24o01-42-users/translation_optimization/trl \
    bash -c "${cmd}"

echo "Training completed!"
echo "Outputs are saved in ${RESULTS_DIR}"
