import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convolution1 = nn.Conv2d(1, 10, kernel_size=5)
        self.convolution2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.convolution2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def accuracy():
    correct = 0.0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    return correct/len(test_loader.dataset)

model = Net()
batch_size = 100
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./temp', train=True, download=True,
                                                          transform=transforms.Compose([transforms.ToTensor()])),
                                                          batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./temp', train=False, download=True,
                                                          transform=transforms.Compose([transforms.ToTensor()])),
                                                          batch_size=batch_size)

optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

print "Accuracy before training: ", accuracy()
for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
print "Accuracy after training: ", accuracy()

# Accuracy before training:  0.1005
# Accuracy after training:  0.8738