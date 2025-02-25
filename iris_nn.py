
import torch
import torch.nn as nn

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

#load dataset
iris = load_iris()
X, y = iris.data, iris.target
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

X_tr_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tr_tensor = torch.tensor(y_train, dtype=torch.long)

#create the neural network
class FullyConnected(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(in_features=4, out_features=64) #input layer to 32 neurons
        self.act1 = nn.ReLU() # relu activation function
        self.l2 = nn.Linear(in_features=64, out_features=16) #32 to 16
        self.drop = nn.Dropout(0.2) #dropout for regularization
        self.act2 = nn.ReLU()
        self.l3 = nn.Linear(16, 3) #output layer to 1 neuron
    
    def forward(self, x):
        #forward pass
        x = self.l1(x)
        x = self.act1(x)
        x = self.l2(x)
        x = self.drop(x)
        x = self.act2(x)
        x = self.l3(x)
        return x

def train_network(model, X_train, y_train):
    epochs = 2000
    loss_arr = []
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.002)
    
    for epoch in range(epochs):
        #make prediction
        y_pred = model(X_train)
        
        #calculate loss
        loss = loss_fn(y_pred, y_train)
        loss_arr.append(loss.item())
        #calculate gradient
        loss.backward()
        
        #update weights
        optim.step()
        optim.zero_grad()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch +1} / {epochs}, loss {loss}")

model = FullyConnected()
train_network(model, X_tr_tensor, y_tr_tensor)

X_test_tr = torch.tensor(X_test, dtype=torch.float32)
y_pred = model(X_test_tr)
y_pred = torch.argmax(y_pred, dim=1)
acc = accuracy_score(y_test, y_pred)
print(acc)