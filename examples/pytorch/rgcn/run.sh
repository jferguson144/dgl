#!/bin/bash
# Baseline with default settings (6k max steps, 100 block regularization, 30k edges sampled per batch)
#CUDA_VISIBLE_DEVICES=0 python -u link_predict.py -d FB15k-237 --gpu 0 | tee output

# Small batch (baseline to check for effect of regularization) block reg w/ 100 blocks
CUDA_VISIBLE_DEVICES=0 python -u link_predict.py -d FB15k-237 --gpu 0 --n-hidden 100 | tee output_small_batch_no_attn

# Small batch without regularization
#CUDA_VISIBLE_DEVICES=1 python -u link_predict.py -d FB15k-237 --gpu 0 --n-bases 1 --n-hidden 100 | tee output_small_batch_no_reg


# Regular batch, multi-headed attention (4). Small batch size + no reg to deal with dimensionality 
#   (input hidden size must be divisible by num_heads*num_blocks. Default blocks for reg is 100, which messes with that)
#CUDA_VISIBLE_DEVICES=2 python -u link_predict.py -d FB15k-237 --gpu 0 --n-bases 1 --n-hidden 100 --n-heads 4 | tee output_4_attn_head
