import numpy as np
import tensorflow as tf

a_data = np.array([[1,1],[1,1]])
b_data = np.array([[2,2],[2,2]])
c_data = np.array([[5,5],[5,5]])
d_data = np.array([[3,3],[3,3]])

a = tf.constant([[1.0,1.0],[1.0,1.0]], name='a')
b = tf.constant([[2.0,2.0],[2.0,2.0]], name='b')
c = tf.Variable(initial_value=[[5.0,5.0],[5.0,5.0]], name='c')
d = tf.placeholder(tf.float32, name='d')
e = (a + b - c) * d

sess = tf.Session()
writer = tf.summary.FileWriter("./temp")
writer.add_graph(sess.graph)

print "Expected:", (a_data + b_data - c_data) * d_data
sess.run(tf.global_variables_initializer())
result = sess.run(e,feed_dict={d:[[3.0,3.0],[3.0,3.0]]})
print "Via Tensorflow: ", result

# Expected: [[-6 -6]
#  [-6 -6]]
# Via Tensorflow:  [[-6 -6]
#  [-6 -6]]