import numpy as np
import tensorflow as tf

a_data = np.array([[1,1],[1,1]])
b_data = np.array([[2,2],[2,2]])
c_data = np.array([[5,5],[5,5]])
d_data = np.array([[3,3],[3,3]])

a = tf.constant([[1,1],[1,1]], name='a')
b = tf.constant([[2,2],[2,2]], name='b')
c = tf.constant([[5,5],[5,5]], name='c')
d = tf.constant([[3,3],[3,3]], name='d')
e = (a + b - c) * d

sess = tf.Session()
writer = tf.summary.FileWriter("./temp")
writer.add_graph(sess.graph)

print "Expected:", (a_data + b_data - c_data) * d_data
result = sess.run(e)
print "Via Tensorflow: ", result

# Expected: [[-6 -6]
#  [-6 -6]]
# Via Tensorflow:  [[-6 -6]
#  [-6 -6]]



