import torch
import numpy

# Converting Torch tensors to Numpy
a1 = torch.ones(5)
print a1.numpy()
# [ 1.  1.  1.  1.  1.]

na1 = numpy.array([1,2,3,4,5])
a2 = torch.from_numpy(na1)
print a2
#  1
#  2
#  3
#  4
#  5
# [torch.LongTensor of size 5]

