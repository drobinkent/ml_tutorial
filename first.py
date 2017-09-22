from tensorflow.examples.tutorials.mnist import input_data
import ssl
import tensorflow as tf
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

res = tf.add(6,8)
sess = tf.Session()
print(sess.run(res))
sess.close()