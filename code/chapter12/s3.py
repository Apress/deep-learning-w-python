import torch

# Concatenation operation along given axis (staring from 0)
a = torch.rand(1,2)
print torch.cat((a,a,a,a),0).size() # (4L, 2L)
print torch.cat((a,a,a,a),1).size() # (1L, 8L)

# Break apart a tensor into parts along
a = torch.rand(10,1)
b = torch.chunk(a,10,0) # 10 parts, axis 0
print len(b) # 10
print b[0].size() # (1L, 1L)

# Get values along an axis given indices (using gather)
t = torch.Tensor([[1,2],[3,4],[5,6]])
print torch.gather(t, 0, torch.LongTensor([[0,0]])) # 1  2
print torch.gather(t, 0, torch.LongTensor([[1,1]])) # 3  4
print torch.gather(t, 0, torch.LongTensor([[2,2]])) # 5  6
print torch.gather(t, 1, torch.LongTensor([[0],[0],[0]])) # 1 3 5
print torch.gather(t, 1, torch.LongTensor([[1],[1],[1]])) # 2 4 6

# Get values along an axis given an indices vector (using index_select)
t = torch.Tensor([[1,2],[3,4],[5,6]])
index = torch.LongTensor([0,0])
print torch.index_select(t, 0, index)
# 1  2
# 1  2
print torch.index_select(t, 1, index)
# 1  1
# 3  3
# 5  5

# Masked Select
t = torch.Tensor([[1,2],[3,4],[5,6]])
mask = t.ge(3) # Greater than
print torch.masked_select(t,mask)
# 3
# 4
# 5
# 6

# Get Indices of Non-Zero elements
print torch.nonzero(torch.Tensor([1, 0, 1, 1, 1]))
#  0
#  2
#  3
#  4
# [torch.LongTensor of size 4x1]

# Split tensor into given size chunks, last one will be smaller
# if dimension of tensor is not not exact multiple of split size
print torch.split(torch.ones(5), split_size=2)
# (
#  1
#  1
# [torch.FloatTensor of size 2]
# ,
#  1
#  1
# [torch.FloatTensor of size 2]
# ,
#  1
# [torch.FloatTensor of size 1]
# )

# Remove dimensions that are of size 1 (essentially dummy)
a = torch.ones(1,3,1,3,1)
print a.size()
# (1L, 3L, 1L, 3L, 1L)
print torch.squeeze(a).size()
# (3L, 3L)

# Remove only the specified dimension
print torch.squeeze(a, dim=0).size()
#(3L, 1L, 3L, 1L)

# Stacking tensors along a dimension
a = torch.rand(1,2)
print torch.stack((a,a,a,a),0).size()
# (4L, 1L, 2L)
print torch.stack((a,a,a,a),1).size()
# (1L, 4L, 2L)
print torch.stack((a,a,a,a),2).size()
# (1L, 2L, 4L)

# Transpose a 2D matrix
a = torch.ones(3,4)
print a.size() # (3L, 4L)
print torch.t(a).size() # (4L, 3L)

# Transpose a matrix
a = torch.ones(3,4,5)
print a.size() # (3L, 4L, 5L)
print torch.transpose(a,0,1).size() # (4L, 3L, 5L)
print torch.transpose(a,0,2).size() # (5L, 4L, 3L)
print torch.transpose(a,1,2).size() # (3L, 5L, 4L)

# Transpose a matrix
a = torch.ones(3,4,5)
print [i.size() for i in torch.unbind(a,0)]
# [(4L, 5L), (4L, 5L), (4L, 5L)]
print [i.size() for i in torch.unbind(a,1)]
# [(3L, 5L), (3L, 5L), (3L, 5L), (3L, 5L)]
print [i.size() for i in torch.unbind(a,2)]
# [(3L, 4L), (3L, 4L), (3L, 4L), (3L, 4L), (3L, 4L)]

# Add a dimension (dummy)
a = torch.ones(3)
print a.size() # (3L,)
print torch.unsqueeze(a, 0).size() # (1L, 3L)
print torch.unsqueeze(a, 1).size() # (3L, 1L)