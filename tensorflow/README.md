# Tensorflow - Example

TensorFlow is a Google-project for developing deep-learning methods. It supports auto-diff and is GPU accelerated. TensorFlow eases the process of defining an objective function in Python. Note, the evaluation is done in a fast C++ backend without the overhead of Python-(GIL).

To minimize `x'Ax + b'x + c`, we just write

```python
x = tf.placeholder(tf.float32, shape=[1, 2], name='input')
A = tf.get_variable('A', dtype=tf.float32, initializer=[[2., 1], [3, 4]], trainable=False)
b = tf.get_variable('b', dtype=tf.float32, initializer=[[1., 1]], trainable=False)
c = tf.get_variable('c', dtype=tf.float32, initializer=[42.], trainable=False)

y = tf.matmul(x, tf.matmul(A, x, transpose_b=True)) + tf.matmul(x, b, transpose_b=True) + c
```

Getting the gradient is as easy as

```python
dx = tf.gradients(y, x)[0]
```

To run this example, you need to compile TensorFlow from source beforehand. Then just run

```console
user@host $ python problem.py
```

This creates the graph definition of the problem and stores the parameters. These information can be loaded in C++ and exploited to get the gradient information easily.

To build this C++ example run

```console
user@host $ # see https://github.com/PatWie/tensorflow_inference for more details
user@host $ export TensorFlow_GIT_REPO=/path/to/tensorflow_git
user@host $ python configure.py
user@host $ cmake .
user@host $ make
```

Solving is done via

```console
user@host $ ./solve
```

or the python way

```console
user@host $ python solve.py
```