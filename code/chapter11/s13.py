import numpy as np
import tensorflow as tf

# Generate Random Data
examples = 1000
features = 100
x_data = np.random.randn(examples, features)
y_data = np.random.randint(size=(examples, 1),low=0, high=2)

# Define the Logistic Regression Model
X = tf.placeholder(tf.float32, shape=[None, features], name = "X")
y = tf.placeholder(tf.float32, shape=[None, 1], name = "y")
w = tf.Variable(tf.random_normal(shape=[features,1]), name= "w")
b = tf.Variable(tf.random_normal(shape=[1,1]), name="b")

# Loss
temp = tf.add(tf.matmul(X, w),b, name="temp")
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=temp, labels=y), name="loss")

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

# Loss before training: 3.50893
# Loss before training: 0.704702
