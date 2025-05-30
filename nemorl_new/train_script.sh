#!/bin/bash

NUM_NODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE

RUN_DPO_PATH=/opt/nemo-rl/examples/run_dpo.py
CONFIG_PATH=/script/dpo.yaml

echo "*******STARTING********"

export PATH="/root/.local/bin:$PATH"
export WANDB_API_KEY="79af9dbae290344f5c04c8069ac6475d3b231866"
source /opt/nemo-rl/.venv/bin/activate

echo "python $RUN_DPO_PATH \
    --config $CONFIG_PATH \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    cluster.num_nodes=$NUM_NODES"

python $RUN_DPO_PATH \
    --config /script/dpo.yaml \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    cluster.num_nodes=$NUM_NODES

echo "*******FINISHING*******"
