import torch
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import random

# Generate training data
data_x = []
data_y = []
max_sequence_len = 10
min_sequence_len = 3
for i in xrange(0,50):
    curr_seq_len = random.randint(min_sequence_len, max_sequence_len)
    # Positive Examples
    data_x.append(torch.ones(curr_seq_len))
    data_y.append(torch.LongTensor([1]))
    # Negative Examples
    temp = torch.ones(curr_seq_len)
    pos = random.randint(0,curr_seq_len-1)
    temp[pos] = 0
    data_x.append(temp)
    data_y.append(torch.LongTensor([0]))

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.inputToHidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.input2output = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.inputToHidden(combined)
        output = self.input2output(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

def accuracy(rnn, data_x, data_y):
    correct = 0.0
    for x, y in zip(data_x, data_y):
        X = Variable(x)
        expected = Variable(y)
        hidden = rnn.initHidden()
        for j in range(len(x)):
            output, hidden = rnn(torch.unsqueeze(X[j],0), hidden)
        pred = output.data.max(1)[1]
        correct += pred.eq(expected.data).sum()
    return correct/len(data_y)


rnn = RNN(1, 1, 2)
hidden = rnn.initHidden()
index = random.randint(0,99)
learning_rate = 0.1
optimizer = torch.optim.SGD(rnn.parameters(), lr = learning_rate)

print "Accuracy before training: ", accuracy(rnn,data_x, data_y)

for i in xrange(500):
    index = random.randint(0,99)
    X = Variable(data_x[index])
    expected = Variable(data_y[index])
    rnn.zero_grad()
    hidden = rnn.initHidden()
    for j in range(len(data_x[index])):
        output, hidden = rnn(torch.unsqueeze(X[j],0), hidden)
    loss = F.cross_entropy(output, expected)
    loss.backward()
    optimizer.step()

print "Accuracy after training: ", accuracy(rnn, data_x, data_y)

# Accuracy before training:  0.5
# Accuracy after training:  0.7