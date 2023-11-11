import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def generate_data():
    num_samples_per_class = 100
    samples = np.random.multivariate_normal(mean=[0,5], cov=[[5,1],[1,0.4]], size=100)
    x = np.array(samples[:,0]).reshape((num_samples_per_class,1)).astype(np.float32)
    y = np.array(samples[:,1]).reshape((num_samples_per_class,1)).astype(np.float32)
    plt.scatter(x, y)
    plt.show()
    return (x,y)

def initialize_weights(input_dim, output_dim):
    # y = b + w1 .x1 
    # W= [w1]
    # X = [x1]
    # b = [b0]
    #   y = activation(dot(input, W) + b)
    # dimensions        (1,1) (1,1)  (1,1)
    W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
    b = tf.Variable(initial_value=tf.random.uniform(shape=(output_dim,)))
    return (W, b)


def model(inputs):
    return tf.matmul(inputs, W) + b


def mean_squared_loss(targets, predictions):
    return tf.reduce_mean(tf.square(targets - predictions))


# training_step will run all the samples in one go i.e == 1 epoch
def training_step(inputs, targets, alpha=0.1):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = mean_squared_loss(targets, predictions)

    gradient_loss_wrt_w, gradient_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(gradient_loss_wrt_w * alpha)
    b.assign_sub(gradient_loss_wrt_b * alpha)
    return loss


def regression_boundary(W, b):
    # w0x1 + w1x2 + b = 0.5 is the equation of regression line
    x = np.linspace(-10, 10, 200)
    y = x * W[0]+ b
    return (x, y)


######################################################

inputs, targets = generate_data()

input_dim = 1
output_dim = 1

W, b = initialize_weights(input_dim, output_dim)
#
#
## start training
for step in range(50):
    loss = training_step(inputs, targets)
    print("loss at step %d:  %f" % (step, loss))
#
## predict
predictions = model(inputs)
#
## Plot linear regression line
x, y = regression_boundary(W, b)
plt.plot(x, y, "r")
plt.scatter(inputs, targets)
plt.show()
