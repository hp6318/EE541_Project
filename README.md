# EE541_Project
EE541 Fall22 Project 

## Title: ASL classification 
### Authors: Meghana Achanta (vachanta@usc.edu); Hardik Prajapati (hprajapa@usc.edu)

#### Demo : Real-time ASL classification

<p align="center"><img src="India.gif" alt="Demo" width="500"/></p>


#### Stack:
```shell 
1) Python
2) Pytorch
3) Google Colab
4) Github
5) Visual Studio Code
6) TensorBoard (plotting learning curves)
```

#### Dataset:
```shell
1) Train Dataset: 87000
2) Test Dataset: 29
3) Classes: 29
4) Image size: 200x200x3
```
#### Pre-processing:
```shell
1) PIL-ToTensor
2) Resize: 200x200 -> 32x32
3) Normalization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
```


#### Best Model Architecture:
```shell
1) LeNET model
2) Optimizer: Adam
3) Criterion: Cross Entropy Loss
4) Initial Learning rate: 0.001
5) Learning rate scheduler: 10% drop after every 2 epochs
6) Total epochs: 5
7) Batch_size: 64
8) Train accuracy: 97%
9) Test accuracy: 100%
```

#### Models Tried:
```shell
Experiment 1: LeNet
Experiment 2: Fully Connected
Experiment 3: LeNet with Data Augmentation (rotation, horizontal flip)
Experiment 4: LeNet with Learning Rate Scheduler (10% drop after every 2 eopchs)
```

#### Engineering Decisions on model Architecture:
```shell 
1) Start with simplest Model, hence MLP with 1 hidden layer(experiment 2). 
2) As dealing with images, try CNN which would learn various filters (edge detection, vertex detection, high pass, low pass, etc.). what's the simplest CNN architecture - LeNET which worked wonders on MNIST dataset. (experiment 1)
3) For real-time, hands wont be stationed. They might be titled a bit in the proposed region. Also, some might use a right hand, some might use a left hand. Hence, Data Augmentation ~ Horizontal Flip, Random Rotation (-5,5)degrees. (experiment 3)
4) It was observed in experiment 1 that, after 2nd epoch, there is considerable increment in train accuracy which indicates that we migh need to slow down the learning rate parameter in order to not jumpy over the optimum point. Hence 10% learning rate drop after every 2 epochs. (experiment 4) 
```

#### Concluding remarks
##### Extensions:
```shell
1) AutoComplete :
    -> To frame the entire sentance with help of autocomplete. 
        eg. If a person signals 'H' 'E', then the system should recommend 'Hello', 'Hey' etc 
```
##### What question do you feel "answered" after this project:
```shell
We code on cpu, Can this be transferred on embedded devices? 
The real-time inference ability of small sized Neural Networks with great accuracy can make it possible to be implemented on embeded devices which come with memory constraints. 
```
##### what are you still curious about after this project:
```shell
Curiosity is whether these trained model would work as good initialization on hand gesture recognition task. Also, will this model work for different skin toned hands? With no plain background? 
```

