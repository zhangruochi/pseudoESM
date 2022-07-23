CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
                --nproc_per_node=1 \
                --nnodes=1          \
                --node_rank=0       \
                --master_addr=localhost  \
                --master_port=22222 \
                 train.py