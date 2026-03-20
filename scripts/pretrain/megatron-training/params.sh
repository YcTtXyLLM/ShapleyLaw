# Pretraining configuration (Transformer-based language model. Following is an example for 1.46B model, please find more configurations for other model sizes in our paper.)

ALL_PARAMS=()

# --------------------------------------------------
# Model architecture
# --------------------------------------------------
ALL_PARAMS+=(
    --num-layers 24
    --hidden-size 2048
    --ffn-hidden-size 7168
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 8

    # Context length
    --seq-length 4096
    --max-position-embeddings 4096

    # Positional encoding
    --position-embedding-type rope
    --rotary-base 500000

    # Architectural choices
    --untie-embeddings-and-output-weights
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --disable-bias-linear
)

# --------------------------------------------------
# Tokenizer
# --------------------------------------------------
ALL_PARAMS+=(
    --tokenizer-type SentencePieceTokenizer
    --tokenizer-model ${TOKENIZER_PATH}
)

# --------------------------------------------------
# Optimization
# --------------------------------------------------
ALL_PARAMS+=(
    --optimizer adam
    --lr 3e-4
    --min-lr 3e-5
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
    --weight-decay 0.1
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
)

# --------------------------------------------------
# Learning rate schedule
# --------------------------------------------------
ALL_PARAMS+=(
    --train-iters ${TRAIN_ITERS}
    --lr-warmup-iters 2000
    --lr-decay-iters ${TRAIN_ITERS}
    --lr-decay-style cosine
)

# --------------------------------------------------
# Batch configuration
# --------------------------------------------------
ALL_PARAMS+=(
    --micro-batch-size 4
    --global-batch-size 512
)

# --------------------------------------------------
# Parallelism and distributed training
# --------------------------------------------------
ALL_PARAMS+=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --distributed-backend nccl
    --distributed-timeout-minutes 120
    --use-mpi
)

# --------------------------------------------------
# Dataset
# --------------------------------------------------
ALL_PARAMS+=(
    --data-path ${TRAIN_DATA_PATH}
    --data-cache-path ${DATA_CACHE_PATH}
)

# --------------------------------------------------
# Implementation details
# --------------------------------------------------
ALL_PARAMS+=(
    --bf16
    --use-mcore-models
    --no-masked-softmax-fusion
    --use-flash-attn
    --attention-softmax-in-fp32
    --transformer-impl transformer_engine
    --attention-backend fused
)
