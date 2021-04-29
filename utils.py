
import numpy as np
import torch.utils.data

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
    def __init__(self, X, transform):
        """x: rows are genes and columns are samples"""
        self.X = X
        self.transform = transform
        
    def __getitem__(self, i):
        x = self.X[i,:]
        return self.transform(x), 0
    
    def __len__(self):
        return len(self.X)
