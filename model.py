import torch
import torch.nn as nn
import torch.nn.functional as F

#LeNet adaptation
#[Conv->relu->max_pool]->[Conv->relu->max_pool]->[Hidden_layer1->relu]->[Hidden_layer2->relu]->Hidden_layer3->Softmax

class Net_1(nn.Module):
  
  def __init__(self):
    super(Net_1,self).__init__()

    self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1,padding=0) #(32-5+1=28) -> ((28-2)/2 + 1 = 14)
    self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0) #(14-5+1=10) -> ((10-2)/2 + 1 = 5) 
    self.hidden1=nn.Linear(in_features=16*5*5,out_features=120)
    self.hidden2=nn.Linear(in_features=120,out_features=84)
    self.hidden3=nn.Linear(in_features=84,out_features=29)
    #nn.softmax(dim=1)

    self.relu=nn.ReLU()
    self.pool=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
  
  def forward(self,x):
    #block1
    x=self.conv1(x)
    x=self.relu(x)
    x=self.pool(x)

    #block2
    x=self.conv2(x)
    x=self.relu(x)
    x=self.pool(x)

    #flatten
    x=torch.flatten(x,1)

    #hidden layer1
    x=self.hidden1(x)
    x=self.relu(x)

    #hidden layer2
    x=self.hidden2(x)
    x=self.relu(x)

    #output layer3
    x=self.hidden3(x)

    return x


#Deep Neural Network adaptation
#[Hidden_layer1->relu]->[Hidden_layer2]->Softmax

class Net_2(nn.Module):
  
  def __init__(self):
    super(Net_2,self).__init__()

    self.hidden1=nn.Linear(in_features=32*32*3,out_features=128)
    self.hidden2=nn.Linear(in_features=128,out_features=29)
    #nn.softmax(dim=1)

    self.relu=nn.ReLU()
    
  def forward(self,x):

    #hidden layer1
    x=self.hidden1(x)
    x=self.relu(x)

    #hidden layer2
    x=self.hidden2(x)

    return x