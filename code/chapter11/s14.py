import numpy as np
import tensorflow as tf

# Generate Random Data
examples = 1000
features = 100
x_data = np.random.randn(examples, features)
y_data = np.random.randn(examples,1)

# Define the Neural Network Model
hidden_layer_nodes = 10
X = tf.placeholder(tf.float32, shape=[None, features], name = "X")
y = tf.placeholder(tf.float32, shape=[None, 1], name = "y")
w1 = tf.Variable(tf.random_normal(shape=[features,hidden_layer_nodes]), name="w1")
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]), name="b1")
w2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,1]), name="w2")
b2 = tf.Variable(tf.random_normal(shape=[1,1]), name="b2")
hidden_output = tf.nn.relu(tf.add(tf.matmul(X, w1), b1), name="hidden_output")
y_hat = tf.nn.relu(tf.add(tf.matmul(hidden_output, w2), b2), name="y_hat")
loss = tf.reduce_mean(tf.square(y_hat - y), name="loss")

# Set up the gradient descent
learning_rate = 0.05
optimiser = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimiser.minimize(loss)

sess = tf.Session()
writer = tf.summary.FileWriter("./temp")
writer.add_graph(sess.graph)
sess.run(tf.global_variables_initializer())

epochs = 5000
batch_size = 5

# Before Training
curr_loss = sess.run(loss, feed_dict={X:x_data, y:y_data})
print "Loss before training:", curr_loss

for i in range(epochs):
    rand_index = np.random.choice(examples, size=batch_size)
    sess.run(train_step, feed_dict={X:x_data[rand_index], y:y_data[rand_index]})

# After Training
curr_loss = sess.run(loss, feed_dict={X:x_data, y:y_data})
print "Loss before training:", curr_loss

# Loss before training: 42.431
# Loss before training: 0.976375