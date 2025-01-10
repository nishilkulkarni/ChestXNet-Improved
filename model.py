import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

class MyChestModel(nn.Module):
    """
    ChestXNet modified model. 
    Pretrained DenseNet121 + sigmoid function at the classifier layer
    """
    def __init__(self, out_classes: int):
        super(MyChestModel, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained = True)
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=out_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.densenet121(x)
        return x
    


