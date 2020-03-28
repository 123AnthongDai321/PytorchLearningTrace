import os
import tensorflow as tf
print("tensoflow version: ", tf.__version__)


a = tf.Constant(2.0)
b = tf.Constant(3.0)

c = a + b

with tf.Session() as sess:
    c_res = sess.run(c)
    print("c_res: ", c_res)