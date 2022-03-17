#!/bin/bash

# Model name: densenet11, densenet21, densenet29 and densenet63
arch=densenet11
# Number of input features
in_features=1686
# We recommend the user to normalize the feature before training the model
data=data/x_new_scanpy_HVG.npz

python main.py \
-a $arch -j 1 -b 256 --lr 0.24 \
--dist-url "tcp://localhost:10001" \
--world-size 1 \
--rank 0 \
--multiprocessing-distributed \
--outdir checkpoint \
--in_features $in_features \
--save_frequency 5 \
--mlp --moco-t 0.2 --moco-k 1024 --moco-m 0.999 --moco-dim 64 $data
