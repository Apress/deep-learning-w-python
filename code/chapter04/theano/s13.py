import numpy
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.printing import pydotprint

def hinge_a(x,y):
    return T.max([0 * x, 1-x*y])

def hinge_b(x,y):
    return ifelse(T.lt(1-x*y,0), 0 * x, 1-x*y)

def hinge_c(x,y):
    return T.switch(T.lt(1-x*y,0), 0 * x, 1-x*y)

x = T.dscalar('x')
y = T.dscalar('y')

z1 = hinge_a(x, y)
z2 = hinge_b(x, y)
z3 = hinge_b(x, y)

f1 = theano.function([x,y], z1)
f2 = theano.function([x,y], z2)
f3 = theano.function([x,y], z3)

pydotprint(f1, outfile="s13-1.png", var_with_name_simple=True)
pydotprint(f2, outfile="s13-2.png", var_with_name_simple=True)
pydotprint(f3, outfile="s13-3.png", var_with_name_simple=True)

print "f(-2, 1) =",f1(-2, 1), f2(-2, 1), f3(-2, 1)
print "f(-1,1 ) =",f1(-1, 1), f2(-1, 1), f3(-1, 1)
print "f(0,1) =",f1(0, 1), f2(0, 1), f3(0, 1)
print "f(1, 1) =",f1(1, 1), f2(1, 1), f3(1, 1)
print "f(2, 1) =",f1(2, 1), f2(2, 1), f3(2, 1)

# f(-2, 1) = 3.0 3.0 3.0
# f(-1,1 ) = 2.0 2.0 2.0
# f(0,1) = 1.0 1.0 1.0
# f(1, 1) = 0.0 0.0 0.0
# f(2, 1) = 0.0 0.0 0.0
