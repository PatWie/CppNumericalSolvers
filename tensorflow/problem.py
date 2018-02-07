import tensorflow as tf
import numpy as np


x = tf.placeholder(tf.float32, shape=[1, 2], name='input')
A = tf.get_variable('A', dtype=tf.float32, initializer=[[2., 1], [3, 4]], trainable=False)
b = tf.get_variable('b', dtype=tf.float32, initializer=[[1., 1]], trainable=False)
c = tf.get_variable('c', dtype=tf.float32, initializer=[42.], trainable=False)

y = tf.matmul(x, tf.matmul(A, x, transpose_b=True)) + tf.matmul(x, b, transpose_b=True) + c
dx = tf.gradients(y, x)[0]

y = tf.identity(y, name='problem_objective')
dx = tf.identity(dx, name='problem_gradient')

# just an example
val = np.array([[1, 1]], dtype=np.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print sess.run(y, {x: val})
    print sess.run(dx, {x: val})

    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, '/tmp/my_problem/my_problem')

    tf.train.write_graph(sess.graph, '/tmp/my_problem', "graph.pb", as_text=False)
