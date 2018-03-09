import numpy as np
import tensorflow as tf

# Construct from diagonal
a = tf.diag([1.0,1.0,1.0], name="a")

# Random Normalised Matrix
b = tf.truncated_normal([3,3], name = "b")

# Simple Fill
c = tf.fill([3,4], -1.0, name = "c")

# Uniform Random
d = tf.random_uniform([3,3], name = "d")

# From Numpy
e = tf.convert_to_tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7., 8.0, 9.0]]), name="e")

# Addition
f = tf.add(a,b, name="f")

# Subtraction
g = tf.subtract(a,b, name="g")

# Multiplcation
h = tf.matmul(a,b, name="h")

# Division
i = tf.transpose(a, name="i")

# Determinant
j = tf.matrix_determinant(d, name = "j")

# Inverse
k = tf.matrix_inverse(e, name = "k")

# Cholesky Decomposition
l = tf.cholesky(a, name = "l")

# Eigen Values and Vectors
m = tf.self_adjoint_eig(a, name = "m")

sess = tf.Session()
writer = tf.summary.FileWriter("./temp")
writer.add_graph(sess.graph)