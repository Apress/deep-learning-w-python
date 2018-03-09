import numpy as np
import tensorflow as tf

# Generate Random Data
examples = 1000
features = 100
x_data = np.random.randn(examples, features)
y_data = np.random.randn(examples,1)

# Define the Linear Regression Model
X = tf.placeholder(tf.float32, shape=[None, features], name = "X")
y = tf.placeholder(tf.float32, shape=[None, 1], name = "y")
w = tf.Variable(tf.random_normal(shape=[features,1]), name= "w")
b = tf.Variable(tf.random_normal(shape=[1,1]), name="b")
y_hat = tf.add(tf.matmul(X,w),b, name="y_hat")

# Define the loss
squared_loss = tf.reduce_sum(tf.pow(y - y_hat,2), name="squared_loss")/examples

# Set up the gradient descent
learning_rate = 0.05
optimiser = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimiser.minimize(squared_loss)

sess = tf.Session()
writer = tf.summary.FileWriter("./temp")
writer.add_graph(sess.graph)
sess.run(tf.global_variables_initializer())

epochs = 5000
batch_size = 5

# Before Training
curr_loss = sess.run(squared_loss, feed_dict={X:x_data, y:y_data})
print "Loss before training:", curr_loss

for i in range(epochs):
    rand_index = np.random.choice(examples, size=batch_size)
    sess.run(train_step, feed_dict={X:x_data[rand_index], y:y_data[rand_index]})

# After Training
curr_loss = sess.run(squared_loss, feed_dict={X:x_data, y:y_data})
print "Loss before training:", curr_loss

# Loss before training: 95.5248
# Loss before training: 2.13263