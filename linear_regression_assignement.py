import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import pandas as pd


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

"Global Variables"
random_number_generator = numpy.random
alpha = 0.1
iteration_number = 200

# Computation graph creation
X = tf.placeholder("float")
Y = tf.placeholder("float")
training_data_size = tf.placeholder("float")
theta_one = tf.Variable(random_number_generator.randn(), name="theta_one")
theta_zero = tf.Variable(random_number_generator.randn(), name="theta_zero")

#theta_one = tf.Variable(0.0, name="theta_one")
#theta_zero = tf.Variable(0.0, name="theta_zero")


# linear predictoin node
y_predicted = tf.add(tf.multiply(X,theta_one ), theta_zero)

# Node for Mean squared error 
error = tf.reduce_sum(tf.pow(y_predicted-Y, 2))/(2*training_data_size)
#Tensorflow optimizer for  linear egression
linear_regression_optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(error)


display_step=5






                    

init = tf.global_variables_initializer()

#n_samples,train_data = read_data()

#my_X = tf.constant(train_data, "float32", shape=[21614,20])
#my_Y = tf.constant(train_data, "float32", shape=[21614,3])
#my_X = numpy.asarray(train_data['sqft_living15'].values)
#my_Y = numpy.asarray(train_data['price'].values)

my_X = numpy.asarray([ 2104,  1600,  2400,  1416,  3000,  1985,  1534,  1427,
  1380,  1494,  1940,  2000,  1890,  4478,  1268,  2300,
  1320,  1236,  2609,  3031,  1767,  1888,  1604,  1962,
  3890,  1100,  1458,  2526,  2200,  2637,  1839,  1000,
  2040,  3137,  1811,  1437,  1239,  2132,  4215,  2162,
  1664,  2238,  2567,  1200,   852,  1852,  1203 ])
my_Y = numpy.asarray([ 399900,  329900,  369000,  232000,  539900,  299900,  314900,  198999,
  212000,  242500,  239999,  347000,  329999,  699900,  259900,  449900,
  299900,  199900,  499998,  599000,  252900,  255000,  242900,  259900,
  573900,  249900,  464500,  469000,  475000,  299900,  349900,  169900,
  314900,  579900,  285900,  249900,  229900,  345000,  549000,  287000,
  368500,  329900,  314000,  299000,  179900,  299900,  239500 ])

size_data_n = normalize(my_X)
price_data_n = normalize(my_Y)

n_samples = my_X.shape[0]


with tf.Session() as sess:
    sess.run(init)

    for i in range(iteration_number):
        for (x, y) in zip(my_X, my_Y):
            sess.run(linear_regression_optimizer, feed_dict={training_data_size: n_samples,X: my_X, Y: my_Y})
            if(i%10 == 0):
                print ("Iteration:", '%04d' % (i + 1), \
                "cost=", "{:.9f}".format(sess.run(error, feed_dict={training_data_size: n_samples,X:size_data_n, Y:price_data_n})),\
            "W=", sess.run(theta_one), "b=", sess.run(theta_zero)  )      