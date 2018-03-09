import theano.tensor as T
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams
import numpy
from theano.printing import pydotprint


random = RandomStreams(seed=42)

a = random.normal((1,3))
b = T.dmatrix('a')

f1 = a * b

g1 = function([b], f1)
pydotprint(g1, outfile="s9.png", var_with_name_simple=True)


print "Invocation 1:", g1(numpy.ones((1,3)))
print "Invocation 2:", g1(numpy.ones((1,3)))
print "Invocation 3:", g1(numpy.ones((1,3)))

# Invocation 1: [[ 1.25614218 -0.53793023 -0.10434045]]
# Invocation 2: [[ 0.66992188 -0.70813926  0.99601177]]
# Invocation 3: [[ 0.0724739  -0.66508406  0.93707751]]
