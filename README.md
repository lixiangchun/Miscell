# Miscell
Mining Information from Single-Cell high-throughput transcriptome data.
Some soure code was copied from [facebookresearch/MoCo](https://github.com/facebookresearch/moco).

The following example was tested with `python-3.9.2` and `torch-1.7.1-cu110`.

### 1. Installation 
```bash
git clone https://github.com/lixiangchun/Miscell.git
cd Miscell

pip install torch
pip install sklearn
pip install scanpy

```

### 2. Training Miscell on own data
The model was trained on GPUs and at least two GPUs are required.

```bash
#!/bin/bash

# Model name: densenet11, densenet21, densenet29 and densenet63
arch=densenet11
# Number of input features
in_features=1686
# We recommend the user to normalize the feature before training the model
data=x_hat_scanpy_HVG.npz

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

```

### 3. Extract feature with pretrained model

```python
import utils
import numpy as np

X = np.load("data/x_new_scanpy_HVG.npz")["x"]
in_features = 1686
assert in_features == X.shape[1]

checkpoint = "checkpoint_0199.pth.tar"

model = utils.load_pretrained_model(arch, in_features, checkpoint, return_feature=True)

features = utils.extract_features(model, X)

```

### 4. Perform DBSCAN clustering on t-SNE features obtained from pretrained model
Install [FIt-SNE](https://github.com/KlugerLab/FIt-SNE).

```python
import sklearn.cluster
import sys
# Modify the path to FIt-SNE accordingly.
sys.path.append('/home/lixc/software/github/FIt-SNE/')
from fast_tsne import fast_tsne

tsne = fast_tsne(features, seed=123, nthreads=12, perplexity_list=[30, 36, 42, 48])
_, y = sklearn.cluster.dbscan(tsne, eps=1, min_samples=5, algorithm='auto')


```

### 5. Perform clustering with scanpy
```python
# https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html#Clustering-the-neighborhood-graph

import numpy as np
import pandas as pd
import scanpy as sc
import anndata

adata=anndata.AnnData(features)
adata.obsm["X_tsne"] = tsne

sc.pp.neighbors(adata, n_neighbors=10, use_rep="X")
sc.tl.leiden(adata)

sc.pl.tsne(adata, color=["leiden"], save='_scanpy.pdf', show=False)


```

### 6. Citation - [Shen et al. Miscell: An efficient self-supervised learning approach for dissecting single-cell transcriptome. iScience 2021(24):11](https://www.sciencedirect.com/science/article/pii/S2589004221011688).
