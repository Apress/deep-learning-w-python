import tensorflow as tf

a = tf.placeholder(tf.float32, name='a')
b = tf.placeholder(tf.float32, name='b')
c = tf.placeholder(tf.float32, name='c')
d = tf.placeholder(tf.float32, name='d')
e = tf.placeholder(tf.float32, name='e')
f = ((a - b + c) * d )/e

sess = tf.Session()
writer = tf.summary.FileWriter("./temp")
writer.add_graph(sess.graph)

print "Expected: ((1 - 2 + 3) * 4)/5.0 = ", ((1 - 2 + 3) * 4)/5.0
result = sess.run(f,feed_dict={a:1,b:2,c:3,d:4,e:5})
print "Via Tensorflow: ((1 - 2 + 3) * 4)/5.0 = ", result

# Expected: ((1 - 2 + 3) * 4)/5.0 =  1.6
# Via Tensorflow: ((1 - 2 + 3) * 4)/5.0 =  1.6

