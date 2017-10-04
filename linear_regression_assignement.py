import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

"Global Variables"
random_number_generator = numpy.random
alpha = 0.01
iteration_number = 1000

# Computation graph creation
X = tf.placeholder("float")
Y = tf.placeholder("float")
training_data_size = tf.placeholder("float")
theta_one = tf.Variable(random_number_generator.randn(), name="theta_one")
theta_zero = tf.Variable(random_number_generator.randn(), name="theta_zero")

# linear predictoin node
y_predicted = tf.add(tf.multiply(X,theta_one ), theta_zero)

# Node for Mean squared error 
error = tf.reduce_sum(tf.pow(y_predicted-Y, 2))/(2*training_data_size)
#Tensorflow optimizer for  linear egression
linear_regression_optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(error)


display_step=5
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(iteration_number):
        for (x, y) in zip(train_X, train_Y):
            sess.run(linear_regression_optimizer, feed_dict={training_data_size: n_samples,X: x, Y: y})

        # Display logs per epoch step
        if (i+1) % display_step == 0:
            c = sess.run(error, feed_dict={training_data_size: n_samples,X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(theta_one), "b=", sess.run(theta_zero))