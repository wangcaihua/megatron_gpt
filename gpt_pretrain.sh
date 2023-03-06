#! /bin/bash

# Runs the "345M" parameter model
export USER=${USER:-wangcaihua}
export DATA_PATH=gpt2_data/my-gpt2_text_document
export CHECKPOINT_PATH=ckpt
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export LOAD=${LOAD:-"--load $CHECKPOINT_PATH"}

# --no-gradient-accumulation-fusion
# --reset-position-ids 
# --reset-attention-mask 
# /data00/home/wangcaihua/miniconda3/lib/python3.10/site-packages/megatron
/data00/home/${USER}/miniconda3/bin/python nlp_pretrain.py \
       --nlp-model-type gpt \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 8 \
       --seq-length 1024 \
       --reset-position-ids \
       --reset-attention-mask \
       --no-masked-softmax-fusion \
       --max-position-embeddings 1024 \
       --train-iters 5000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --split 949,50,1 \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 ${LOAD} $@

