import torch
import numpy

# Create a Tensor
a1 = torch.Tensor(4,3)
print a1
# 1.00000e-35 *
#   0.0000  0.0000  0.0000
#   0.0000  0.0004  0.0000
#   0.0104  0.0000  0.0156
#   0.0000  1.5086  0.0000
# [torch.FloatTensor of size 4x3]

# Create a Tensor populated with random values
a2 = torch.rand(4,3)
print a2
#  0.6553  0.7280  0.5829
#  0.5965  0.1383  0.8214
#  0.7690  0.7348  0.2798
#  0.6695  0.4295  0.2672
# [torch.FloatTensor of size 4x3]

# Normalised (0 mean, unit (1) variance)
a3 = torch.randn(5)
print a3
# -0.9593
# -2.2416
#  0.5279
# -0.4319
#  1.4821
# [torch.FloatTensor of size 5]

# Digonal Matrices
a4 = torch.eye(3)
print a4
#  1  0  0
#  0  1  0
#  0  0  1
# [torch.FloatTensor of size 3x3]

# From Numpy
a5 = torch.from_numpy(numpy.array([1,2,3]))
print a5
#  1
#  2
#  3
# [torch.LongTensor of size 3]

# Linearly spaced
a6 = torch.linspace(0,1,steps=5)
print a6
#  0.0000
#  0.2500
#  0.5000
#  0.7500
#  1.0000
# [torch.FloatTensor of size 5]

# Logarithmically spaced
a7 = torch.logspace(1,3,steps=3)
print a7
#    10
#   100
#  1000
# [torch.FloatTensor of size 3]

# Ones and Zeros
a8 = torch.ones(5)
print a8
#  1
#  1
#  1
#  1
#  1
# [torch.FloatTensor of size 5]

a9 = torch.zeros(5)
print a9
#  0
#  0
#  0
#  0
#  0
# [torch.FloatTensor of size 5]

# Random Permutation of numbers from 0 to n-1
a10 = torch.randperm(5)
print a10
#  3
#  1
#  2
#  4
#  0
# [torch.LongTensor of size 5]

# Range from start to end with given step
a11 = torch.arange(1,10,step=2)
print a11
#  1
#  3
#  5
#  7
#  9
# [torch.FloatTensor of size 5]

