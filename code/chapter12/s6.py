import torch
from torch.autograd import Variable
import torch.nn as nn
import random
import torch.optim as optim

# Define the Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)
    def forward(self, x):
        return self.linear(x)

def rmse(lr_model, x, y):
    y_hat = []
    for i in x:
        y_hat.append(lr_model(i))
    Y = torch.stack(y, 1)
    Y_hat = torch.stack(y_hat, 1)
    diff = Y - Y_hat
    return torch.sqrt(torch.mean(diff * diff))

# Prepare Dataset
dataset_size = 100
x_data = []
y_data = []
for i in xrange(0,100):
    x = Variable(torch.rand(1,1))
    y = Variable(torch.rand(1,1))
    x_data.append(x)
    y_data.append(y)

lr = LinearRegression(1)

loss_func = torch.nn.MSELoss()
optimizer = optim.SGD(lr.parameters(), lr = 0.01, momentum=0.9)

print "RMSE before training ", rmse(lr, x_data, y_data).data.numpy()

# Training
steps = 5000
for i in xrange(steps):
    index = random.randint(0, dataset_size-1)
    lr.zero_grad()
    loss = loss_func(lr(x_data[index]), y_data[index])
    loss.backward()
    optimizer.step()

print "RMSE after training ", rmse(lr, x_data, y_data).data.numpy()

# RMSE before training  [ 0.57391912]
# RMSE after training  [ 0.31023207]