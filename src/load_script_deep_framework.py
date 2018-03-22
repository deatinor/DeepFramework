import torch
import math

# Import default load script from parent folder

def load_dataset():
    train=torch.rand([1000,2])
    target=train.norm(p=2,dim=1)
    target=target<1/math.sqrt(2*math.pi)

    return train,target

