import torch
from torch import FloatTensor, LongTensor
import math

from load_script_deep_framework import load_dataset
from framework import *

# Load train and test dataset
train,target_train = load_dataset()
test,target_test = load_dataset()

# Define the Neural Network
layers = [
    Linear(2,25),
    ReLU(),
    Linear(25,25),
    ReLU(),
    Linear(25,25),
    ReLU(),
    Linear(25,2),
    Tanh()
]
loss_function = LossMSE()
network = Sequential(layers, loss_function)

# Initial learning rate and decay rate and optimizer
learning_rate = 2e-2
decay_rate = 0.98
optimizer=OptimSGD(network, learning_rate, decay_rate)


num_epochs = 200
mini_batch_size = 5


for epoch in range(num_epochs):
    
    # Randomize order for SGD
    permutation = torch.randperm(target_train.shape[0])
    
    # Split train dataset in batches of dimension mini_batch_size
    train_batched = [train[permutation][i:(i+mini_batch_size),:] for i in range(train.shape[0]) if i % mini_batch_size == 0]
    target_batched = [target_train[permutation][i:(i+mini_batch_size),:] for i in range(train.shape[0]) if i % mini_batch_size == 0]
    
    # Loop over the batches for training
    for j, batch in enumerate(zip(train_batched, target_batched)):
        
        # Load sample
        train_element, target_element = batch
        
        # Forward pass
        output = network(train_element)
        loss = loss_function(output,target_element)

        # Backward pass, gradient step and reinitialize gradient
        network.backward()

        optimizer.step()
        optimizer.zero_grad_()
    
    # Multiply learning rate by decay rate
    optimizer.learning_rate_decay_step()



# Compute train and test errors
compute_error(network, train, target_train, data_type='Train')
compute_error(network, test, target_test, data_type='Test')


