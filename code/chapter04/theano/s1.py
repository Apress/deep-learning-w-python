import theano.tensor as T
from theano import function
from theano.printing import pydotprint

a = T.dscalar('a')
b = T.dscalar('b')
c = T.dscalar('c')
d = T.dscalar('d')
e = T.dscalar('e')

f = ((a - b + c) * d )/e

g = function([a, b, c, d, e], f)

print "Expected: ((1 - 2 + 3) * 4)/5.0 = ", ((1 - 2 + 3) * 4)/5.0
print "Via Theano: ((1 - 2 + 3) * 4)/5.0 = ",  g(1, 2, 3, 4, 5)

pydotprint(g, outfile="s1.png", var_with_name_simple=True)

# Expected: ((1 - 2 + 3) * 4)/5.0 =  1.6
# Via Theano: ((1 - 2 + 3) * 4)/5.0 =  1.6

