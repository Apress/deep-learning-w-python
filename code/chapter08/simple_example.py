import os
os.environ["THEANO_FLAGS"]="device=cpu,floatX=float64,exception_verbosity=high"

import downhill
import theano
import theano.tensor as T
import numpy
import pylab

import sklearn.datasets
train_X, train_y = sklearn.datasets.make_moons(5000, noise=0.1)
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

batch_size = 1
z1 = X.dot(W1) + b1
a1 = T.tanh(z1)
z2 = a1.dot(W2) + b2
y_hat = T.nnet.softmax(z2)
loss_reg = 1./batch_size * reg_lambda/2 * (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2)))
loss = T.nnet.categorical_crossentropy(y_hat, y).mean() + loss_reg

prediction = T.argmax(y_hat, axis=1)
predict = theano.function([X], prediction)

train_loss = []
validation_loss = []

opt = downhill.build('adadelta', loss=loss)
train = downhill.Dataset([train_X[:-1000], train_y_onehot[:-1000]], batch_size=batch_size, iteration_size=1)
valid = downhill.Dataset([train_X[-1000:], train_y_onehot[-1000:]])
iterations = 0
for tm, vm in opt.iterate(train, valid, patience=1000):
    iterations += 1
    train_loss.append(tm['loss'])
    validation_loss.append(vm['loss'])
    if iterations > 5000:
            break

x_min, x_max = train_X[:, 0].min() - 0.5, train_X[:, 0].max() + 0.5
y_min, y_max = train_X[:, 1].min() - 0.5, train_X[:, 1].max() + 0.5
x_mesh, y_mesh = numpy.meshgrid(numpy.arange(x_min, x_max, 0.01), numpy.arange(y_min, y_max, 0.01))
Z = predict(numpy.c_[x_mesh.ravel(), y_mesh.ravel()])
Z = Z.reshape(x_mesh.shape)
pylab.contourf(x_mesh, y_mesh, Z, cmap=pylab.cm.Spectral)
pylab.scatter(train_X[:-1000, 0], train_X[:-1000, 1], c=train_y[:-1000], cmap=pylab.cm.Spectral)