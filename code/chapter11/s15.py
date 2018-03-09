import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Helper function to flatten a tensor for convolution and pooling operations
def flatten_out(X, height, width, channels,
                convolution_output_height_dim,
                convolution_output_width_dim,
                stride, padding):

    # Pad zeros
    X_padded = tf.pad(X, [[0,0], [padding, padding], [padding, padding], [0,0]])

    # Simulate the sliding of the convolution weights (filter) as a window over the images
    slices = []
    for i in range(convolution_output_height_dim):
        for j in range(convolution_output_width_dim):
            window = tf.slice(X_padded, [0, i*stride, j*stride, 0], [-1, height, width, -1])
            slices.append(window)

    # Combine, reshape and return result
    stacked = tf.stack(slices)
    return tf.reshape(stacked, [-1, channels * width * height])

# Convolution Operation
def convolution(X, conv_weights,
                conv_bias, padding,
                stride, name="convolution"):

    with tf.name_scope(name):

        # Extract dimensions of input (X)
        # and convolution weights
        X_filter_count_dim, \
        X_height_dim, \
        X_width_dim, \
        X_channels_dim = [d.value for d in X.get_shape()]

        cw_height_dim, \
        cw_width_dim, \
        cw_channels_dim, \
        cw_filter_count_dim = [d.value for d in conv_weights.get_shape()]

        # Compute the output dimensions of the
        # of the convolution operation on
        # X and conv_weights
        convolution_output_height_dim =\
            (X_height_dim + 2*padding - cw_height_dim)//stride + 1

        convolution_output_width_dim =\
            (X_width_dim + 2*padding - cw_width_dim)//stride + 1

        # Flatten X and conv_weights so that a
        # matrix mutiplication will lead
        # to a convolution operation
        X_flattened = flatten_out(X, cw_height_dim,
                                  cw_width_dim, cw_channels_dim,
                                  convolution_output_height_dim,
                                  convolution_output_width_dim,
                                  stride, padding)

        cw_flattened = tf.reshape(conv_weights, [cw_height_dim *
                                                 cw_width_dim *
                                                 cw_channels_dim,
                                                 cw_filter_count_dim])

        # Multiply the flattened matrices
        z = tf.matmul(X_flattened, cw_flattened) + conv_bias

        # Unflatten/reorganise and return result
        return tf.transpose(tf.reshape(z, [convolution_output_height_dim,
                                           convolution_output_width_dim,
                                           X_filter_count_dim,
                                           cw_filter_count_dim]),
                            [2, 0, 1, 3])

# ReLU operation
def relu(X, name = "relu"):
    with tf.name_scope(name):
        return tf.maximum(X, tf.zeros_like(X))

# Max Pooling Operation
def max_pooling(X, pooling_height, pooling_width,
                padding, stride, name ="pooling"):
    with tf.name_scope(name):
        # Get dimensions of input (X)
        X_filter_count_dim, \
        X_height_dim, \
        X_width_dim, \
        X_channels_dim = [d.value for d in X.get_shape()]

        # Compute the output dimensions of the result
        # of the convolution operation on
        # X and conv_weights
        convolution_output_height_dim = (X_height_dim + 2 * padding - pooling_height) // stride + 1
        convolution_output_width_dim = (X_width_dim + 2 * padding - pooling_width) // stride + 1

        # Flatten for max operation
        X_flattened = flatten_out(X, pooling_height, pooling_width,
                                  X_channels_dim,
                                  convolution_output_height_dim,
                                  convolution_output_width_dim, stride, padding)
        # Max Pooling
        pool = tf.reduce_max(tf.reshape(X_flattened,
                                        [convolution_output_height_dim,
                                         convolution_output_width_dim,
                                         X_filter_count_dim,
                                         pooling_height *
                                         pooling_width,
                                         X_channels_dim]),
                             axis=3)
        # Reorg and return result
        return tf.transpose(pool, [2, 0, 1, 3])

# Fully connected layer
def fully_connected(X, W, b, name="fully-connected"):
    with tf.name_scope(name):
        n = X.get_shape()[0].value
        X_flat = tf.reshape(X, [n, -1])
        return tf.matmul(X_flat, W) + b

# Softmax
def softmax(X, name="softmax"):
    with tf.name_scope(name):
        X_centered = X - tf.reduce_max(X)
        X_exp = tf.exp(X_centered)
        exp_sum = tf.reduce_sum(X_exp, axis=1)
        return tf.transpose(tf.transpose(X_exp) / exp_sum)

# Cross Entropy (Loss function for training)
def cross_entropy(y, t, name="cross-entropy"):
    with tf.name_scope(name):
        return -tf.reduce_mean(tf.log(tf.reduce_sum(y * t, axis=1)))

# Accuracy (for evalution)
def accuracy(network, t, name="accuracy"):
    with tf.name_scope(name):
        t_predict = tf.argmax(network, axis=1)
        t_actual = tf.argmax(t, axis=1)
        return tf.reduce_mean(tf.cast(tf.equal(t_predict, t_actual), tf.float32))

# Read the input
mnist = input_data.read_data_sets("./temp", one_hot=True, reshape=False)

# Parameters describing the input data
batch_size = 1000
image_height = 28
image_width = 28
image_channels = 1 # monochromatic images
categories = 10

# Placeholders for input and output
X = tf.placeholder(tf.float32, shape=[batch_size, image_height, image_width, image_channels], name = "X")
y = tf.placeholder(tf.float32, shape=[batch_size, categories], name = "y")

# Convolution weight parameters
conv_height = 7
conv_width = 7
conv_channels = 1
conv_filter_count = 20

# Convolution weight and bias
convolution_weights = tf.Variable(tf.random_normal([conv_height, conv_width, conv_channels, conv_filter_count], stddev=0.01),
                                  name="convolution_weights")
convolution_bias = tf.Variable(tf.zeros([conv_filter_count]), name="convolution_bias")

# Convolution Layer
conv_layer = convolution(X, convolution_weights, convolution_bias, padding=2, stride=1, name="Convolution")

# Convolution Layer Activation
conv_activation_layer = relu(conv_layer,name="convolution_actiavation_relu")

# Pooling Layer
pooling_layer = max_pooling(conv_activation_layer, pooling_height=2, pooling_width=2, padding=0, stride=2, name ="Pooling")

# Fully Connected Layer-1 (Hidden Layer)
hidden_size = 150
batch_size, pool_output_h, pool_output_w, conv_filter_count = [d.value for d in pooling_layer.get_shape()]
weights1 = tf.Variable(tf.random_normal([pool_output_h * pool_output_w * conv_filter_count, hidden_size], stddev=0.01),
                       name="weights1")
bias1 = tf.Variable(tf.zeros([hidden_size]), name="bias1")
fully_connected1 = fully_connected(pooling_layer, weights1, bias1, name="Fully-Connected-1")
fully_connected1_activation = relu(fully_connected1, name="fully_connected1_activation_relu")

# Fully Connected Layer-2 (Output Layer)
output_size = 10
weights2 = tf.Variable(tf.random_normal([hidden_size, output_size], stddev=0.01), name="weights2")
bias2 = tf.Variable(tf.zeros([output_size]), name="bias2")
fully_connected2 = fully_connected(fully_connected1_activation, weights2, bias2, name="Fully-Connected-2")

# Softmax
softmax_layer = softmax(fully_connected2, name="Softmax")

# Cross Entropy Loss
loss = cross_entropy(softmax_layer, y, name ="Cross-Entropy")

# Training and Evaluation
learning_rate = 0.1
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Before Training
test_x = mnist.test.images[:batch_size]
test_y = mnist.test.labels[:batch_size]

sess = tf.Session()
writer = tf.summary.FileWriter("./temp")
writer.add_graph(sess.graph)
sess.run(tf.global_variables_initializer())

print "Accuracy before training:", sess.run(accuracy(softmax_layer, y), feed_dict={X:test_x, y:test_y})
batches = int(mnist.train.num_examples/batch_size)
steps = 5
for i in xrange(steps):
    for j in xrange(batches):
        x_data, y_data = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={X:x_data, y:y_data})
print "Accuracy after training:", sess.run(accuracy(softmax_layer, y), feed_dict={X:test_x, y:test_y})

# Accuracy before training: 0.124
# Accuracy after training: 0.894