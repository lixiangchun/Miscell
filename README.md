# Miscell
Mining Information from Single-Cell high-throughput transcriptome data


### 1. Training Miscell on own data
As an example, we provided a preprocessed dataset on Baidu Disk (link:xxx) (extraction code: xxxxx)
```{bash}
arch=densenet12
in_features=1506
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
```{python, eval=False}
import utils
import numpy as np

X = np.load("example.npz")["x"]
checkpoint = "checkpoint_0199.pth.tar"

model = utils.load_pretrained_model(arch, in_features, checkpoint)

features = utils.extract_features(model, X)

```

