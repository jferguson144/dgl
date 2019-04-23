#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -u link_predict.py -d FB15k-237 --gpu 0 | tee output
