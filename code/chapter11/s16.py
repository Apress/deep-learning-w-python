import tensorflow as tf

W = tf.Variable([.3], tf.float32, name="W")
b = tf.Variable([-.3], tf.float32, name = "b")
x = tf.placeholder(tf.float32, name="x")
with tf.name_scope("linear_model"):
    linear_model = W * x + b

    # Tensorboard Histogram Summary of W and b
    tf.summary.histogram("W", W)
    tf.summary.histogram("b", b)

y = tf.placeholder(tf.float32, name="y")
with tf.name_scope("loss_computation"):
    loss = tf.reduce_sum(tf.square(linear_model - y))

    # Tensorboard Scalar Summaries for Loss
    tf.summary.scalar("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
init = tf.global_variables_initializer()

# Merge Summaries
merged_summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(init)

# Graph Summary
writer = tf.summary.FileWriter("./temp")
writer.add_graph(sess.graph)

for i in range(1000):
  train_result, summary_result = sess.run([train,merged_summary], {x:x_train, y:y_train})
  writer.add_summary(summary_result, i)

curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

# W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11

