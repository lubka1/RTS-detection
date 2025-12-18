#!/bin/bash

module purge
module load cuda/12.2

conda activate rts-tf

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH



# chmod +x ~/activate_tf_gpu.sh
# use in any session
# source ~/activate_tf_gpu.sh
