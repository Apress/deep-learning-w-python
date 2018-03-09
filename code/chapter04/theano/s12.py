import numpy
import theano
import theano.tensor as T
import sklearn.metrics
from theano.printing import pydotprint

def l2(x):
    return T.sum(x**2)

examples = 1000
features = 100
hidden = 10

D = (numpy.random.randn(examples, features), numpy.random.randint(size=examples, low=0, high=2))
training_steps = 1000

x = T.dmatrix("x")
y = T.dvector("y")

w1 = theano.shared(numpy.random.randn(features, hidden), name="w1")
b1 = theano.shared(numpy.zeros(hidden), name="b1")

w2 = theano.shared(numpy.random.randn(hidden), name="w2")
b2 = theano.shared(0., name="b2")

p1 = T.tanh(T.dot(x, w1) + b1)
p2 = T.tanh(T.dot(p1, w2) + b2)

prediction = p2 > 0.5

error = T.nnet.binary_crossentropy(p2,y)

loss = error.mean() + 0.01 * (l2(w1) + l2(w2))
gw1, gb1, gw2, gb2 = T.grad(loss, [w1, b1, w2, b2])

train = theano.function(inputs=[x,y],outputs=[p2, error], updates=((w1, w1 - 0.1 * gw1), (b1, b1 - 0.1 * gb1), (w2, w2 - 0.1 * gw2), (b2, b2 - 0.1 * gb2)))
predict = theano.function(inputs=[x], outputs=[prediction])

pydotprint(predict, outfile="s12.png", var_with_name_simple=True)

print "Accuracy before Training:", sklearn.metrics.accuracy_score(D[1], numpy.array(predict(D[0])).ravel())

for i in range(training_steps):
     prediction, error = train(D[0], D[1])

print "Accuracy after Training:", sklearn.metrics.accuracy_score(D[1], numpy.array(predict(D[0])).ravel())

# Accuracy before Training: 0.51
# Accuracy after Training: 0.716
