import tensorflow as tf

a = tf.placeholder(tf.float32, name='a')
b = tf.placeholder(tf.float32, name='b')
c = tf.constant(3.0, name='c')
d = tf.constant(4.0, name='d')
e = tf.constant(5.0, name='e')
f = ((a - b + c) * d )/e

sess = tf.Session()
writer = tf.summary.FileWriter("./temp")
writer.add_graph(sess.graph)

print "Expected: ((1 - 2 + 3) * 4)/5.0 = ", ((1 - 2 + 3) * 4)/5.0
result = sess.run(f,feed_dict={a:1,b:2})
print "Via Tensorflow: ((1 - 2 + 3) * 4)/5.0 = ", result

# Expected: ((1 - 2 + 3) * 4)/5.0 =  1.6
# Via Tensorflow: ((1 - 2 + 3) * 4)/5.0 =  1.6

