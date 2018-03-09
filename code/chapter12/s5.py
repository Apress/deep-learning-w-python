import torch
from torch.autograd import Variable
import torch.nn as nn
import random

# Define the Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.w = Variable(torch.rand(1,1), requires_grad = True)
        self.b = Variable(torch.rand(1,1), requires_grad = True)
    def forward(self, x):
        return (x * self.w) + self.b

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
print "RMSE before training ", rmse(lr, x_data, y_data).data.numpy()

# Training
steps = 1000
learning_rate = 0.0001
for i in xrange(steps):
    index = random.randint(0, dataset_size-1)
    lr.w.data.zero_()
    lr.b.data.zero_()
    loss = loss_func(lr(x_data[index]), y_data[index])
    loss.backward()
    lr.w.data -= learning_rate * lr.w.grad.data
    lr.b.data -= learning_rate * lr.b.grad.data

print "RMSE after training ", rmse(lr, x_data, y_data).data.numpy()

# RMSE before training  [ 0.73697698]
# RMSE after training  [ 0.49001607]