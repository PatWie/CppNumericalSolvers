import tensorflow as tf

x = tf.get_variable('x', dtype=tf.float32, initializer=[[10., 1]], trainable=True)
x = tf.identity(x)

with tf.Session() as sess:

    # load the computation graph
    loader = tf.train.import_meta_graph('/tmp/my_problem/my_problem.meta', input_map={"input:0": x})
    loader = loader.restore(sess, '/tmp/my_problem/my_problem')

    y = tf.get_default_graph().get_tensor_by_name('problem_objective:0')

    # opt = tf.train.GradientDescentOptimizer(0.001).minimize(y)
    opt = tf.train.AdamOptimizer(0.01).minimize(y)
    sess.run(tf.global_variables_initializer())


    for i in range(10000):
        l, _ = sess.run([y, opt])

    print sess.run(y)
    print sess.run(x)