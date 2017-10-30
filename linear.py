import tensorflow as tf
sess = tf.Session()
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

## initialize variables
init = tf.global_variables_initializer()
sess.run(init)

## excecute the linear model
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))


"""
A loss function measures how far apart the current model is from the provided data.
We'll use a standard loss model for linear regression,
which sums the squares of the deltas between the current model and the provided data.
linear_model - y creates a vector where each element is the
corresponding example's error delta.
We call tf.square to square that error. Then, we sum all the
squared errors to create a single scalar that abstracts the error
of all examples using tf.reduce_sum:
"""
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

## now using the perfect values for W and b
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
## we got error 0
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
valW, valB = sess.run([W, b])
print("W: %s b: %s "%(valW,valB))