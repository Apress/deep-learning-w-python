import theano.tensor as T
from theano import function
from theano.printing import pydotprint

# L1 Regularization
def l1(x):
    return T.sum(abs(x))

# L2 Regularization
def l2(x):
    return T.sum(x**2)

a = T.dmatrix('a')
f_a = l1(a)
f_l1 = function([a], f_a)
print "L1 Regularization:", f_l1([[0,1,3]])
pydotprint(f_l1, outfile="s8-1.png", var_with_name_simple=True)

b = T.dmatrix('b')
f_b = l2(b)
f_l2 = function([b], f_b)
print "L2 Regularization:", f_l2([[0,1,3]])
pydotprint(f_l2, outfile="s8-2.png", var_with_name_simple=True)

# L1 Regularization: 4.0
# L2 Regularization: 10.0
