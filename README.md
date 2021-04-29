# Miscell
Mining Information from Single-Cell high-throughput transcriptome data.
Some soure code was copied from [facebookresearch/MoCo](https://github.com/facebookresearch/moco).


### 1. Training Miscell on own data
As an example, we provided a preprocessed dataset on [Baidu Disk](https://pan.baidu.com/s/1QfdWEsoqFxhnFwqhlNKlsw) (extraction code: acpq).

```{bash}
# Model name: densenet11, densenet21, densenet29 and densenet63
arch=densenet11
# Number of input features
in_features=3186
# We recommend the user to normalize the feature before training the model
data=expr_orignal/x_hat.npz

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


### 2. Extract feature with pretrained model
```{python}
import utils
import numpy as np

X = np.load("example.npz")["x"]
checkpoint = "checkpoint_0199.pth.tar"

model = utils.load_pretrained_model(arch, in_features, checkpoint, return_feature=True)

features = utils.extract_features(model, X)

```


### 3. Perform clustering with DBSCAN algorithm
Install [FIt-SNE]().

```{python}
import sklearn
import sklearn.cluster
sys.path.append('/home/lixc/software/github/FIt-SNE/')
from fast_tsne import fast_tsne

tsne = fast_tsne(features, seed=123, nthreads=12, perplexity_list=[30, 36, 42, 48])
_, y = sklearn.cluster.dbscan(tsne, eps=1, min_samples=5, algorithm='auto')


```

### 4. Perform clustering with scannpy
```{python}

```

