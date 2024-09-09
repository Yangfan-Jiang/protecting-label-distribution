# Machine learning models
import torch
from torch import nn
from kymatio.torch import Scattering2D


class multiClassHuberLoss(nn.Module):
    def __init__(self, h=0.1):
        super(multiClassHuberLoss, self).__init__()
        self.h = h

    def huber_fn(self, y, t):
        t1 = (y*t > 1+self.h) * torch.tensor(0.0, requires_grad=True)
        t2 = (-self.h <= 1-y*t) * (1-y*t <= self.h) * (1+self.h-y*t)**2 / (4.0*self.h)
        t3 = (y*t < 1-self.h) * (1-y*t)
        return (t1+t2+t3).sum(1)
        
    def forward(self, output, y): #output: batchsize * n_class
        return self.huber_fn(output, y)


def get_scatter_transform():
    shape = (28, 28, 1)
    scattering = Scattering2D(J=2, shape=shape[:2])
    K = 81 * shape[2]
    (h, w) = shape[:2]
    return scattering, K, (h//4, w//4)


class ScatterLinear(nn.Module):
    """
    ScatterNet model used in the following paper
    - Tramer, Florian, and Dan Boneh. Differentially Private Learning Needs Better Features (or Much More Data). In ICLR 2021. 
    See https://github.com/ftramer/Handcrafted-DP/blob/main/models.py
    """
    def __init__(self, in_channels, hw_dims, input_norm=None, classes=10, clip_norm=None, **kwargs):
        super(ScatterLinear, self).__init__()
        self.K = in_channels
        self.h = hw_dims[0]
        self.w = hw_dims[1]
        self.fc = None
        self.norm = None
        self.clip = None
        self.build(input_norm, classes=classes, clip_norm=clip_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, clip_norm=None, classes=10):
        self.fc = nn.Linear(self.K * self.h * self.w, classes)

        if input_norm is None:
            self.norm = nn.Identity()
        elif input_norm == "GroupNorm":
            self.norm = nn.GroupNorm(num_groups, self.K, affine=False)
        else:
            self.norm = lambda x: standardize(x, bn_stats)

        if clip_norm is None:
            self.clip = nn.Identity()
        else:
            self.clip = ClipLayer(clip_norm)

    def forward(self, x):
        x = self.norm(x.view(-1, self.K, self.h, self.w))
        x = self.clip(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
class ScatterLinearSVM(nn.Module):
    """
    Identical to ScatterLinear. Use a different loss function.
    """
    def __init__(self, in_channels, hw_dims, input_norm=None, classes=10, clip_norm=None, **kwargs):
        super(ScatterLinearSVM, self).__init__()
        self.K = in_channels
        self.h = hw_dims[0]
        self.w = hw_dims[1]
        self.fc = None
        self.norm = None
        self.clip = None
        self.build(input_norm, classes=classes, clip_norm=clip_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, clip_norm=None, classes=10):
        self.fc = nn.Linear(self.K * self.h * self.w, classes)

        if input_norm is None:
            self.norm = nn.Identity()
        elif input_norm == "GroupNorm":
            self.norm = nn.GroupNorm(num_groups, self.K, affine=False)
        else:
            self.norm = lambda x: standardize(x, bn_stats)

        if clip_norm is None:
            self.clip = nn.Identity()
        else:
            self.clip = ClipLayer(clip_norm)

    def forward(self, x):
        x = self.norm(x.view(-1, self.K, self.h, self.w))
        x = self.clip(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    
class LogisticRegression(nn.Module):
    """Logistic regression"""
    def __init__(self, num_feature, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_feature, output_size)

    def forward(self, x):
        return self.linear(x)

