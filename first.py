import tensorflow as tf



x = [1, 2, 3]
y = [4, 5]
zipped = zip(x, y)
print(zipped)

for x in zipped : 
    print(x)


x = tf.read_file("boston_housing.csv")
with tf.Session() as sess:
    sess.run(x)
    print(x.get_shape())


    # Display logs per epoch step
        *if (i+1) % display_step == 0:
            c = sess.run(error, feed_dict={training_data_size: n_samples,X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(theta_one), "b=", sess.run(theta_zero))