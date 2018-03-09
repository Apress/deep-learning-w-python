import theano.tensor as T
from theano import function
from theano.printing import pydotprint

# sigmoid
a = T.dmatrix('a')
f_a = T.nnet.sigmoid(a)
f_sigmoid = function([a],[f_a])
print "sigmoid:", f_sigmoid([[-1,0,1]])
pydotprint(f_sigmoid, outfile="s4-1.png", var_with_name_simple=True)

# tanh
b = T.dmatrix('b')
f_b = T.tanh(b)
f_tanh = function([b],[f_b])
print "tanh:", f_tanh([[-1,0,1]])
pydotprint(f_tanh, outfile="s4-2.png", var_with_name_simple=True)

# fast sigmoid
c = T.dmatrix('c')
f_c = T.nnet.ultra_fast_sigmoid(c)
f_fast_sigmoid = function([c],[f_c])
print "fast sigmoid:", f_fast_sigmoid([[-1,0,1]])
pydotprint(f_fast_sigmoid, outfile="s4-3.png", var_with_name_simple=True)

# softplus
d = T.dmatrix('d')
f_d = T.nnet.softplus(d)
f_softplus = function([d],[f_d])
print "soft plus:",f_softplus([[-1,0,1]])
pydotprint(f_softplus, outfile="s4-4.png", var_with_name_simple=True)

# relu
e = T.dmatrix('e')
f_e = T.nnet.relu(e)
f_relu = function([e],[f_e])
print "relu:",f_relu([[-1,0,1]])
pydotprint(f_relu, outfile="s4-5.png", var_with_name_simple=True)

# softmax
f = T.dmatrix('f')
f_f = T.nnet.softmax(f)
f_softmax = function([f],[f_f])
print "soft max:",f_softmax([[-1,0,1]])
pydotprint(f_softmax, outfile="s4-6.png", var_with_name_simple=True)
