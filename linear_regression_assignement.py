import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import pandas as pd


"Global Variables"
random_number_generator = numpy.random
alpha = 0.5
iteration_number = 250
display_step=5


def normalize(array):
    return (array - array.mean()) / array.std()

"""
Data used here for learnign is kaggle housing data
all the columns are follwoing 
['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15']
We are applying gradiant descent on price as Y and sqft_living15 as X
"""
def read_data():
    all_data = pd.read_csv("kaggle_housing_data.csv")
    print(all_data.columns)
    #print(all_data.size)
    number_of_columns = all_data.columns.size
    total_sample_size = (all_data.size+1)/number_of_columns
    return total_sample_size , all_data



# Computation graph creation
X = tf.placeholder("float")
Y = tf.placeholder("float")
training_data_size = tf.placeholder("float")
theta_one = tf.Variable(100.0, name="theta_one")
theta_zero = tf.Variable(200.0, name="theta_zero")

# linear predictoin node
y_predicted = tf.add(tf.multiply(X,theta_one ), theta_zero)

# Node for Mean squared error 
error = tf.reduce_sum(tf.pow(y_predicted-Y, 2))/(2*training_data_size)
#Tensorflow optimizer for  linear egression
linear_regression_optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(error)

     


#n_samples,train_data = read_data()

#my_X = tf.constant(train_data, "float32", shape=[21614,20])
#my_Y = tf.constant(train_data, "float32", shape=[21614,3])
#my_X = numpy.asarray(train_data['sqft_living15'].values)
#my_Y = numpy.asarray(train_data['price'].values)

my_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
my_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])



n_samples = my_X.shape[0]

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for i in range(iteration_number):
        for (x, y) in zip(my_X, my_Y):
            sess.run(linear_regression_optimizer, feed_dict={training_data_size: n_samples,X: x, Y: y})
            if i%10 == 0:
                print ("Iteration:", '%04d' % (i + 1), \
                "cost=", "{:.9f}".format(sess.run(error, feed_dict={training_data_size: n_samples,X:my_X, Y:my_Y})),\
                "W=", sess.run(theta_one), "b=", sess.run(theta_zero)  )      
