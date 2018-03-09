import theano.tensor as T
from theano import function
from theano import shared
from theano.printing import pydotprint

import numpy

x = T.dmatrix('x')
y = shared(numpy.array([[4, 5, 6]]))
z = T.sum(((x * x) + y) * x)

f = function(inputs = [x], outputs = [z])

g = T.grad(z,[x])
g_f = function([x], g)

pydotprint(f, outfile="s6.png", var_with_name_simple=True)

print "Original:", f([[1, 2, 3]])
print "Original Gradient:", g_f([[1, 2, 3]])

y.set_value(numpy.array([[1, 1, 1]]))
print "Updated:", f([[1, 2, 3]])
print "Updated Gradient", g_f([[1, 2, 3]])

# Original: [array(68.0)]
# Original Gradient: [array([[  7.,  17.,  33.]])]
# Updated: [array(42.0)]
# Updated Gradient [array([[  4.,  13.,  28.]])]
