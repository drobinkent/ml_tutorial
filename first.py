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