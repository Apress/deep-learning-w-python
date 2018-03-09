import numpy as np
import tensorflow as tf

a = tf.placeholder(tf.float32, name='a')
b = tf.placeholder(tf.float32, name='b')
c = tf.Variable(initial_value=[[5.0,5.0],[5.0,5.0]], name='c')
d = tf.Variable(initial_value=[[3.0,3.0],[3.0,3.0]], name='d')

p = tf.placeholder(tf.float32, name='p')
q = tf.placeholder(tf.float32, name='q')
r = tf.Variable(initial_value=3.0, name='r')
s = tf.Variable(initial_value=4.0, name='s')
u = tf.constant(5.0, name='u')
e = (((a * p) + (b - q) - (c + r )) * d/s) * u

sess = tf.Session()
writer = tf.summary.FileWriter("./temp")
writer.add_graph(sess.graph)

a_data = np.array([[1,1],[1,1]])
b_data = np.array([[2,2],[2,2]])
c_data = np.array([[5,5],[5,5]])
d_data = np.array([[3,3],[3,3]])
print "Expected:", (((a_data * 1.0) + (b_data - 2.0) - (c_data + 3.0 )) * d_data/4.0) * 5.0
sess.run(tf.global_variables_initializer())
result = sess.run(e,feed_dict={p:1.0, q:2.0, a:[[1,1],[1,1]],b:[[2,2],[2,2]]})
print "Via Tensorflow: ", result

# Expected: [[-26.25 -26.25]
#  [-26.25 -26.25]]
# Via Tensorflow:  [[-26.25 -26.25]
#  [-26.25 -26.25]]


