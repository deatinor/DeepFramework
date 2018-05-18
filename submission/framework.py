import torch
from torch import FloatTensor, LongTensor
import math

class Module():
    '''General class structure from which to inherit'''
    def forward(self,input):
        raise NotImplemented
        
    def backward(self,input):
        raise NotImplemented
            
    def param(self):
        return []
    
    def __call__(self, *input):
        return self.forward(*input)


class Linear(Module):
    '''Linear layer implementation, needs the number of inputs and number of units to call an instance.
    It initializes Gaussian weights with mean = 0 and std = 0.1.
    Bias terms are included in the weights matrix.'''

    def __init__(self, input_dimension, output_dimension):
        super(Linear,self).__init__()
        
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
    
        self._weights = torch.randn(self._output_dimension, self._input_dimension + 1)*1e-1

        self._gradient = torch.zeros(self._weights.shape)
    
    def forward(self, input):
        
        self._input = input.view(-1, self._input_dimension)

        # Append '1' to input to be multiplied to bias term
        self._input = torch.cat((self._input, torch.Tensor(self._input.shape[0],1).fill_(1)), dim=1)

        self._output = self._input.mm(self._weights.t())
        
        return self._output.clone()
        
    def backward(self, d_dy):
        self._gradient.add_(d_dy.t().mm(self._input))
        
        # Narrowing is done to exclude bias terms in backprop
        d_dx = d_dy.mm(self._weights.narrow(1, 0, self._input_dimension))
        
        return d_dx
    
    def param(self):
        return [self._weights, self._gradient]


class ReLU(Module):
    '''ReLU activation function class, it performs backward and forward pass.'''
    def __init__(self):
        super(ReLU,self).__init__()
        
    def forward(self,input):
        self._input = input.clone()
        
        self._output = self._input.clone()
        self._output[self._output < 0] = 0

        return self._output.clone()
    
    def backward(self,d_dy):
        d_dx = d_dy.clone()
        d_dx[self._input < 0] = 0
        
        return d_dx


class Tanh(Module):
    '''Tanh activation function class, it performs backward and forward pass.'''
    def __init__(self):
        super(Tanh,self).__init__()
        
    def forward(self,input):
        self._input = input.clone()
        
        self._output = self._input.tanh()
        
        return self._output.clone()
    
    def backward(self,d_dy):
        d_dx= (1 - self._input.tanh()**2).mul(d_dy)
        
        return d_dx
        

class LossMSE(Module):
    def __init__(self):
        super(LossMSE,self).__init__()
        
    def forward(self, input, target):
        '''Returns square error between input and target.'''
        self._input = input - target
        self._output = (self._input).pow(2).sum()

        return self._output
        
    def backward(self):
        d_dy = 2 * self._input
        return d_dy


class OptimSGD():
    def __init__(self,sequential, learning_rate, decay_rate=1):
        super(OptimSGD,self).__init__()
        
        self._learning_rate = learning_rate
        self._sequential= sequential
        self._decay_rate = decay_rate
    
    def step(self):
        '''Computes an update step. Each parameter is a tuple if (parameter, gradient)'''
        for param in self._sequential.param():
            if param:
                param[0].add_(- self._learning_rate * param[1]) 
        
    def zero_grad_(self):
        for param_grad in self._sequential.param():
            if param_grad:
                param_grad[1].zero_()
        
    def learning_rate_decay_step(self):
        self._learning_rate *= self._decay_rate
        

class Sequential(Module):
    '''Builds a neural network, takes a list of layers and activation functions + loss function when instantiated.
    Performs the forward pass for the input through all the layers and returns loss.
    Similarly performs backward pass.'''
    
    def __init__(self, modules ,loss):
        super(Sequential,self).__init__()
        
        self._modules = modules
        self._loss = loss
    
    def forward(self, input):
        
        y = input.clone()
        for module in self._modules:
            y = module(y)
        
        return y
    
    def backward(self):
        d_dy = self._loss.backward()
        
        for module in reversed(self._modules):
            d_dy = module.backward(d_dy)
    
    def param(self):
        
        param_list=[]
        
        for module in self._modules:
            if module.param():
                param_list.append(module.param())
                
        return param_list


def compute_error(network, dataset, target, data_type='Train'):

    correct = 0

    # Forward pass
    output = network(dataset)

    boolean_target = target[:,1] > target[:,0]
    boolean_output = output[:,1] > output[:,0]

    # Count number of correct predictions
    correct += sum(boolean_output == boolean_target)

    correct = (1000-correct)/1000 * 100
    print(data_type + ' error: {0:.2f} %'.format(correct))

    return correct


###################################################################################

'Other functions'

import matplotlib.pyplot as plt
from matplotlib import colors

'Dataset plotting function'
def plot_output(train, all_output, permutation):
    plt.figure(figsize=(3,3))
    plt.scatter(train[permutation][:,0], train[permutation][:,1], c = all_output)
