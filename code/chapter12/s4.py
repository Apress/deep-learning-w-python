import torch
from torch.autograd import Variable

def f(x, y, w, b):
    y_hat = torch.tanh(torch.dot(x,w) + b)
    return torch.sqrt((y_hat - y) * (y_hat - y))

x = Variable(torch.rand(5), requires_grad=False)
y = Variable(torch.rand(1,1), requires_grad=False)
w = Variable(torch.rand(5), requires_grad=True)
b = Variable(torch.ones(1,1),requires_grad=True)
result = f(x, y, w, b)
result.backward()
print w.grad
# Variable containing:
# 1.00000e-02 *
#   5.7800
#   2.4759
#   1.8131
#   3.8120
#   0.5258
# [torch.FloatTensor of size 5]
print b.grad
# Variable containing:
# 1.00000e-02 *
#   6.6010
# [torch.FloatTensor of size 1x1]