import torch
import math

# Import default load script from parent folder

def load_dataset():
    train=torch.rand([1000,2])
    target=torch.ones([1000,2])
    target_bool=train.norm(p=2,dim=1)
    target_bool=target_bool<1/math.sqrt(2*math.pi)
    target[:,0][target_bool==1]=-1
    target[:,1][target_bool==0]=-1

    return train,target

