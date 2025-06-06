#!/bin/bash
#SBATCH --job-name=dpo_fine_tuning
#SBATCH --output=dpo_test_run_output.txt
#SBATCH --error=dpo_test_run_error.txt
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
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


NUM_NODES=$SLURM_JOB_NUM_NODES      # Total number of nodes you are using
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE   # Number of GPUs on each node
MASTER_ADDR=${HOSTS[0]}             # IP address (acrually, the name for which dns is configured) of the MASTER node (first node in the list)
MASTER_PORT=29500                   # A free port on the MASTER node for communication

# Calculate total number of processes/GPUs
TOTAL_PROCESSES=$(($NUM_NODES * $GPUS_PER_NODE))

# Container path
CONTAINER=/ceph/hpc/data/s24o01-42-users/containers/transformers_deepspeed_latest.sif

# Path to your training script
TRAINING_SCRIPT="train.py"

# Get the parameters for the training script
DPO_RANK=${1:-64}
LEARNING_RATE=${2:-4e-7}
BETA=${3:-0.2}

# SSL Certificate Configuration
CACERT_FILE_PATH="/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/cacert.pem"
export SSL_CERT_FILE="${CACERT_FILE_PATH}"
export REQUESTS_CA_BUNDLE="${CACERT_FILE_PATH}"

# Training script arguments
SCRIPT_ARGS="--rank=${DPO_RANK} --learning_rate=${LEARNING_RATE} --total_epochs=3 --beta=${BETA}"

# Accelerate config file path
ACCELERATE_CONFIG_FILE=/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/accelerate_config.yaml

# Set NCCL debug to info for more detailed logging if issues persist
# export NCCL_DEBUG=INFO

# Export wandb key
export WANDB_API_KEY="79af9dbae290344f5c04c8069ac6475d3b231866"

# Preparing the results directory and output files
RESULTS_DIR="/ceph/hpc/data/s24o01-42-users/translation_optimization/trl/result_dir"
OUTFILE="${RESULTS_DIR}/rm-%j_%t.out"
ERRFILE="${RESULTS_DIR}/rm-%j_%t.err"
mkdir -p ${RESULTS_DIR}

# Writing the launch command using accelerate in the cmd variable
read -r -d '' cmd <<EOF
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
    --deepspeed_hostfile "$HOSTFILE_PATH" \
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
    --deepspeed_hostfile "$HOSTFILE_PATH" \
    --deepspeed_multinode_launcher "standard" \
    --rdzv_backend static \
    --same_network \
    --mixed_precision bf16 \
    "$TRAINING_SCRIPT" \
    $SCRIPT_ARGS \
&& echo "--- Script finished on Node Rank: \$SLURM_PROCID ---"
EOF

# Execute the command
srun \
    --cpu-bind=verbose \
    --job-name=${SLURM_JOB_NAME} \
    --nodes=${SLURM_JOB_NUM_NODES} \
    --ntasks-per-node=1 \
    --gpus-per-task=$GPUS_PER_NODE \
    --output=$OUTFILE \
    singularity exec \
        --nv \
        $CONTAINER \
        bash -c "${cmd}"

echo "Training completed!"
echo "Outputs are saved in ${RESULTS_DIR}"
