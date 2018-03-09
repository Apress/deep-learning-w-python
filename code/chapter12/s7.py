import torch
from torch.autograd import Variable
import torch.nn as nn
import random
import torch.optim as optim
import torch.nn.functional as F

# Define NN Model
class NNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NNModel, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, 2)
    def forward(self, x):
        temp = self.linear1(x)
        temp = F.relu(temp)
        temp = self.linear2(temp)
        temp = F.softmax(temp)
        return temp

def accuracy(nn_model, x_data, y_data):
    correct = 0.0
    for x, y in zip(x_data, y_data):
        output = nn_model(x)
        pred = output.data.max(1)[1]
        temp = pred.eq(y.data)
        curr_correct = temp.sum()
        correct += curr_correct
    return correct/len(y_data)

# Prepare Dataset
dataset_size = 100
x_data = []
y_data = []
for i in xrange(0,dataset_size):
    x = Variable(torch.rand(1,5))
    if random.random() > 0.5:
        y = Variable(torch.LongTensor([0]))
    else:
        y = Variable(torch.LongTensor([1]))
    x_data.append(x)
    y_data.append(y)

nnet = NNModel(5,20)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(nnet.parameters(), lr = 0.01)

print "Accuracy before training ", accuracy(nnet, x_data, y_data)

# Training
steps = 10000
for i in xrange(steps):
    index = random.randint(0, dataset_size-1)
    nnet.zero_grad()
    loss = loss_func(nnet(x_data[index]), y_data[index])
    loss.backward()
    optimizer.step()

print "Accuracy after training ", accuracy(nnet, x_data, y_data)

# Accuracy before training  0.455
# Accuracy after training  0.62

