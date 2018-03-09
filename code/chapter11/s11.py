import tensorflow as tf

# Creating a tensor
t1 = tf.zeros([1,20], name="t1")

# Creating variables
v1 = tf.Variable(t1, name="v1")
v2 = tf.Variable(t1, name="v2")

# Creating variables based on given dimensions
r = 4
c = 5
t2 = tf.zeros([r,c], name="t2")
t3 = tf.ones([r,c], name="t3")
v3 = tf.Variable(t2, name="v3")
v4 = tf.Variable(t3, name="v4")

# Using the shape of a previously defined variable
v5 = tf.Variable(tf.zeros_like(v3), name="v5")
v6 = tf.Variable(tf.ones_like(v4), name="v6")

# Fill Initialization
v7 = tf.Variable(tf.fill([r, c], -42), name="v7")

# Constant Initialization
v8 = tf.Variable(tf.constant([1,2,3,4,5,6,7,8]), name="v8")

# Constant Initialization
v9 = tf.Variable(tf.constant(42, shape=[r, c]), name="v9")

# Linearly spaced Initialization
v10 = tf.Variable(tf.linspace(start=-10.0, stop=10.0, num=100), name="v10")

# Range Initialization
v11 = tf.Variable(tf.range(start=-1.0, limit=1, delta=0.1), name="v11")

# Random Normal Initialization
v12 = tf.random_normal([r, c], mean=0.0, stddev=1.0, name="v12")

# Add the graph for visualization on TensorBoard
sess = tf.Session()
writer = tf.summary.FileWriter("./temp")
writer.add_graph(sess.graph)
