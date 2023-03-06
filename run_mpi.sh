
export USER=wangcaihua
export PIPELINE_PARALLEL_SIZE=2
export TENSOR_PARALLEL_SIZE=2
export DATA_PARALLEL_SIZE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export LOAD=" "
CMD=${1:-gpt_pretrain.sh}

# --hostfile ~/hostfile.txt
# --load $CHECKPOINT_PATH \
/data00/shared/mpirun --map-by ppr:4:node -np 4 --tag-output \
       -mca pml ob1 -mca btl ^openib \
       -bind-to none -map-by slot \
       -x NCCL_IB_DISABLE=1 \
       -x MASTER_ADDR=localhost \
       -x MASTER_PORT=9876 \
       -x CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
       -x PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE} \
       -x TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} \
       --wdir /home/${USER}/code/megatron_gpt \
       ${CMD} --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
       --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE}
       