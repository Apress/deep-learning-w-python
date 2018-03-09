import os
os.environ["THEANO_FLAGS"]="device=cpu,floatX=float64,exception_verbosity=high"

import downhill
import sklearn.datasets

import theano
import theano.tensor as T
import numpy
import pylab

import sklearn.datasets
train_X, train_y = sklearn.datasets.make_moons(5000, noise=0.01)
train_y_onehot = numpy.eye(2)[train_y]

numpy.random.seed(0)
num_examples = len(train_X)
reg_lambda = numpy.float64(0.01)
nn_input_dim = 2
nn_hdim = 1000
nn_output_dim = 2

W1_val = numpy.random.randn(nn_input_dim, nn_hdim)
b1_val = numpy.zeros(nn_hdim)
W2_val = numpy.random.randn(nn_hdim, nn_output_dim)
b2_val = numpy.zeros(nn_output_dim)

X = T.matrix('X')
y = T.matrix('y')
W1 = theano.shared(W1_val, name='W1')
b1 = theano.shared(b1_val, name='b1')
W2 = theano.shared(W2_val, name='W2')
b2 = theano.shared(b2_val, name='b2')

batch_size = 100
z1 = X.dot(W1) + b1
a1 = T.tanh(z1)
z2 = a1.dot(W2) + b2
y_hat = T.nnet.softmax(z2)
loss_reg = 1./batch_size * reg_lambda/2 * (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2)))
loss = T.nnet.categorical_crossentropy(y_hat, y).mean() + loss_reg

prediction = T.argmax(y_hat, axis=1)
forward_prop = theano.function([X], y_hat)
calculate_loss = theano.function([X, y], loss)
predict = theano.function([X], prediction)

def build_model(algo, b):
    loss_value = []
    global batch_size

    W1.set_value(W1_val)
    b1.set_value(b1_val)
    W2.set_value(W2_val)
    b2.set_value(b2_val)

    batch_size = b
    opt = downhill.build('sgd', loss=loss)

    train = downhill.Dataset([train_X[:-1000], train_y_onehot[:-1000]], batch_size=b, iteration_size=1)
    valid = downhill.Dataset([train_X[-1000:], train_y_onehot[-1000:]])
    iterations = 0
    for tm, vm in opt.iterate(train, valid, patience=1000):
        iterations += 1
        loss_value.append(vm['loss'])
        if iterations > 1000:
            break
    return loss_value

def plot_decision_boundary(pred_func, X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    pylab.cla()
    pylab.clf()
    pylab.contourf(xx, yy, Z, cmap=pylab.cm.Spectral)
    pylab.scatter(X[:, 0], X[:, 1], c=y, cmap=pylab.cm.Spectral)
    pylab.savefig("/Users/nikhil.ketkar/Desktop/show.png")


# full_batch = build_model('sgd', len(train_X[:-1000]))
# stocastic = build_model('sgd', 1)
# mini_match = build_model('sgd', 10)
# adagrad = build_model('adagrad', 1)
# adagrad = build_model('adadelta', 1)


def build_model(algo):
    loss_value = []

    W1.set_value(W1_val)
    b1.set_value(b1_val)
    W2.set_value(W2_val)
    b2.set_value(b2_val)

    opt = downhill.build(algo, loss=loss)

    train = downhill.Dataset([train_X[:-1000], train_y_onehot[:-1000]], batch_size=1, iteration_size=1)
    valid = downhill.Dataset([train_X[-1000:], train_y_onehot[-1000:]])
    iterations = 0
    for tm, vm in opt.iterate(train, valid, patience=1000):
        iterations += 1
        loss_value.append(vm['loss'])
        if iterations > 1000:
            break
    return loss_value

algo_names = ['adadelta', 'adagrad', 'adam', 'nag', 'rmsprop', 'rprop', 'sgd']
losses = []
for algo_name in algo_names:
    print algo_name
    vloss = build_model(algo_name)
    losses.append(numpy.array(vloss))

# 'esgd
for l in losses:
    pylab.plot(l)
pylab.legend(algo_names)
pylab.savefig("/Users/nikhil.ketkar/Desktop/compare.pdf")

# pylab.savefig("/Users/nikhil.ketkar/Desktop/compare.png")
# plot_decision_boundary(predict,train_X,train_y)