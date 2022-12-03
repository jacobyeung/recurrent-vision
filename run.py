# Import libraries for convolutional neural network and data processing
# We will use pytorch to create a convolutional neural network training on MNIST dataset. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Import the MNIST dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from basic_CNN import CNN

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_dataloaders():
    train_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transforms.ToTensor())
    test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transforms.ToTensor())
    test_data_noisy = datasets.MNIST(root = 'data', train = False, download = True, 
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(0., 1.)
    ]))

    return {
        'train' : torch.utils.data.DataLoader(train_data, 
                                            batch_size=512, 
                                            shuffle=True,),
        
        'test'  : torch.utils.data.DataLoader(test_data, 
                                            batch_size=512, 
                                            shuffle=True,),

        'test_data_noisy'  : torch.utils.data.DataLoader(test_data_noisy, 
                                            batch_size=512, 
                                            shuffle=True,),
    }

def train(num_epochs, cnn, loaders, device):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.01)
    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images).to(device)   # batch x
            b_y = Variable(labels).to(device)   # batch y
            # b_x = images
            # b_y = labels
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

def test(loaders, label, device):
    # Test the model
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders[label]:
            images = images.to(device)
            labels = labels.to(device)
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / float(total) 
        print('Test Accuracy of the model on the 10000 test images: %.4f' % accuracy)

cnn = CNN(num_layers=2, num_recurrence=1, num_channels=8).to(device)
loaders = get_dataloaders()
train(10, cnn, loaders, device)
test(loaders, 'test', device)
test(loaders, 'test_data_noisy', device)