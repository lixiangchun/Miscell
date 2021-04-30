import numpy as np
import torch.utils.data
from densenet import *

class TwoCropsTransform:
    """Take two random crops of one input as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        #k = self.base_transform(x)
        #return [q, k]
        return [q, x]
    
class RandomSubArrayShuffle(object):
    def __init__(self, ratio=0.1):
        self.ratio = ratio
        
    def __call__(self, x):
        n = len(x)
        idxs = np.array(range(n))
        k = np.int(np.floor(n * self.ratio))
        
        idxs1 = np.random.choice(idxs, k, replace=False)
        idxs2 = np.random.choice(np.setdiff1d(idxs, idxs1), k, replace=False)
        
        a = np.copy(x)
        a[idxs1] = x[idxs2]
        
        return a

class RandomZero(object):
    def __init__(self, ratio=0.1):
        self.ratio = ratio
        
    def __call__(self, x):
        n = len(x)
        idxs = np.array(range(n))
        k = np.int(np.floor(n * self.ratio))
        
        idxs = np.random.choice(idxs, k, replace=False)
        
        a = np.copy(x)
        a[idxs] = 0
        
        return a

class RandomGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.001):
        self.mean = mean
        self.std = std
        
    def __call__(self, x):
        return x + np.random.normal(loc=0.0, scale=self.std, size=len(x))

class MoCoDataset(torch.utils.data.Dataset):
    def __init__(self, X, transform=None):
        """x: rows are genes and columns are samples"""
        self.X = X
        self.transform = transform
        
    def __getitem__(self, i):
        x = self.X[i,:]
        if self.transform:
            x = self.transform(x)
        return x, 0
    
    def __len__(self):
        return len(self.X)

def get_model(arch, in_features, num_classes=1000, return_feature=False):
    model = None

    kwargs = {'num_classes': num_classes, 'return_feature':return_feature}
    if arch == 'densenet11':
        model = densenet11(in_features, **kwargs)
    elif arch == 'densenet63':
        model = densenet63(in_features, **kwargs)
    elif arch == 'densenet21':
        model = densenet21(in_features, **kwargs)
    elif arch == 'densenet29':
        model = densenet29(in_features, **kwargs)
    else:
        raise ValueError("Unknown arch {}".format(arch))

    return model

def load_pretrained_model(arch, in_features, checkpoint, num_classes=1000, return_feature=True):
    model = get_model(arch, in_features, num_classes=num_classes, return_feature=return_feature)

    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    checkpoint = torch.load(checkpoint, map_location="cpu")
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        
    return model

def extract_features(model, X):
    # set to eval mode
    model.eval()
    
    val_dataset = MoCoDataset(X)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256, shuffle=False,
        num_workers=2)

    features = []
    
    with torch.no_grad():
        for x, _ in val_loader:
            output = model(x)
            features.extend(output.detach().cpu().numpy())
            
    features = np.asarray(features)
    
    return features


