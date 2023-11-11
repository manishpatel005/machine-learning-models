import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def generate_data():
    num_samples_per_class = 1000
    negative_samples = np.random.multivariate_normal(
        mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
    )
    positive_samples = np.random.multivariate_normal(
        mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
    )
    inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
    targets = np.vstack(
        (np.zeros((num_samples_per_class, 1)), np.ones((num_samples_per_class, 1)))
    ).astype(np.float32)
    plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
    plt.show()
    return (inputs, targets)


def initialize_weights(input_dim, output_dim):
    # y = b + w1 .x1 + w2.x2
    # W= [w1,
    #     w2]
    # X = [x1, x2]
    # b = [b0]
    #   y = activation(dot(input, W) + b)
    # dimensions        (1,2) (2,1)  (1,1)
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


def classifier_boundary(W, b):
    # w0x1 + w1x2 + b = 0.5 is the equation of classifier line
    x = np.linspace(-4, 6, 200)
    y = (0.5 - b) / W[1] - x * W[0] / W[1]
    return (x, y)


######################################################

inputs, targets = generate_data()

input_dim = 2
output_dim = 1

W, b = initialize_weights(input_dim, output_dim)


# start training
for step in range(50):
    loss = training_step(inputs, targets)
    print("loss at step %d:  %f" % (step, loss))

# predict
predictions = model(inputs)

# plot the predictions
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()

# Plot classifier boundary
x, y = classifier_boundary(W, b)
plt.plot(x, y, "r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
