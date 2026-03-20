# --------------------------------------------------
# Training setup
# --------------------------------------------------

# Number of training iterations
TRAIN_ITERS=$(cat ${CONFIG_DIR}/train_iters.txt)

# Training data configuration
source ${CONFIG_DIR}/train_data.sh

# Model hyperparameters (defines ALL_PARAMS)
# Requires TRAIN_ITERS and TRAIN_DATA_PATH
source ${CONFIG_DIR}/params.sh

# --------------------------------------------------
# Logging configuration (optional)
# --------------------------------------------------

# It is recommended to login to the tracking service externally
# (e.g., via environment variable or CLI)
# Example: wandb login <API_KEY>

WANDB_ENTITY=${WANDB_ENTITY:-"anonymous"}
WANDB_PROJECT=${WANDB_PROJECT:-"project"}
WANDB_JOB_NAME="pretrain-${TASK_NAME}"

ALL_PARAMS+=(
    --log-interval 1
    --log-throughput
    --wandb-entity ${WANDB_ENTITY}
    --wandb-project ${WANDB_PROJECT}
    --wandb-exp-name ${WANDB_JOB_NAME}
)

# --------------------------------------------------
# Checkpointing
# --------------------------------------------------

CHECKPOINT_DIR=${OUTPUT_DIR}/checkpoints

ALL_PARAMS+=(
    --load ${CHECKPOINT_DIR}
    --save ${CHECKPOINT_DIR}
    --save-interval 1000
)

# --------------------------------------------------
# Debug (optional)
# --------------------------------------------------

echo "Training parameters:"
echo "${ALL_PARAMS[@]}"

# --------------------------------------------------
# Distributed training launch
# --------------------------------------------------

mpirun \
    --display-allocation \
    --report-bindings \
    --oversubscribe \
    -np ${NUM_GPUS} \
    --npernode ${NUM_GPUS_PER_NODE} \
    -bind-to none \
    -map-by slot \
    python \
        ${TRAIN_SCRIPT_PATH} \
        ${ALL_PARAMS[@]}
