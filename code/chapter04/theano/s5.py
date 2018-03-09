import theano.tensor as T
from theano import function
from theano import shared
from theano.printing import pydotprint


import numpy

x = T.dmatrix('x')
y = shared(numpy.array([[4, 5, 6]]))
z = x + y
f = function(inputs = [x], outputs = [z])

pydotprint(f, outfile="s5.png", var_with_name_simple=True)

print "Original Shared Value:", y.get_value()
print "Original Function Evaluation:", f([[1, 2, 3]])

y.set_value(numpy.array([[5, 6, 7]]))

print "Original Shared Value:", y.get_value()
print "Original Function Evaluation:", f([[1, 2, 3]])

# Couldn't import dot_parser, loading of dot files will not be possible.
# Original Shared Value: [[4 5 6]]
# Original Function Evaluation: [array([[ 5.,  7.,  9.]])]
# Original Shared Value: [[5 6 7]]
# Original Function Evaluation: [array([[  6.,   8.,  10.]])]
