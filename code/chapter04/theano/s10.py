import numpy
import theano
import theano.tensor as T
import sklearn.metrics
from theano.printing import pydotprint

def l2(x):
    return T.sum(x**2)

examples = 1000
features = 100

D = (numpy.random.randn(examples, features), numpy.random.randint(size=examples, low=0, high=2))
training_steps = 1000

x = T.dmatrix("x")
y = T.dvector("y")
w = theano.shared(numpy.random.randn(features), name="w")
b = theano.shared(0., name="b")

p = 1 / (1 + T.exp(-T.dot(x, w) - b))
error = T.nnet.binary_crossentropy(p,y)
loss = error.mean() + 0.01 * l2(w)
prediction = p > 0.5
gw, gb = T.grad(loss, [w, b])

train = theano.function(inputs=[x,y],outputs=[p, error], updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

print "Accuracy before Training:",sklearn.metrics.accuracy_score(D[1], predict(D[0]))

for i in range(training_steps):
    prediction, error = train(D[0], D[1])

print "Accuracy before Training:", sklearn.metrics.accuracy_score(D[1], predict(D[0]))

pydotprint(predict, outfile="s10.png", var_with_name_simple=True)

# Accuracy before Training: 0.481
# Accuracy before Training: 0.629
