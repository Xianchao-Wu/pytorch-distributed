#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 -H localhost:4 --verbose python 5.2.horovod_pytorch_mnist.py --fp16-allreduce 
