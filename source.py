# PART A
import tensorflow as tf
import numpy as np
# for visuals
import tqdm
import matplotlib.pyplot as plt


def filterClasses(x, y):
    """
    Keeps only classes 5 and 6
    :param x: dataset features
    :param y: dataset labels
    :return: the dataset only with filtered classes
    """
    keep = (y == 5) | (y == 6)
    x, y = x[keep], y[keep]
    y = y == 5
    return x, y


def reshapeToVector(train):
    """
    function to reshape images from 28x28 to a 784 vector
    :param train: dataset to reshape
    :return: a numpy array with the reshapes dataset and with type float64
    """
    resh_train = []
    for i in range(len(train)):
        resh_train.append(np.reshape(np.array(train[i]), 784).astype("float64"))
    return np.array(resh_train)


def rescale(train):
    """
    Rescales dataset [0,255] -> [0,1]
    :param train: dataset
    :return: the rescaled dataset
    """
    for i in tqdm.tqdm(range(len(train))):
        for j in range(len(train[i])):
            if train[i][j] == 0:
                continue
            train[i][j] = train[i][j] / 255.0


# a)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, y_train = filterClasses(x_train, y_train)
x_test, y_test = filterClasses(x_test, y_test)
print("Loading: DONE")

# create validation set from train set
(x_val, y_val) = (x_train[int(0.8 * len(x_train)):], y_train[int(0.8 * len(y_train)):])
(x_train, y_train) = (x_train[:int(0.8 * len(x_train))], y_train[:int(0.8 * len(y_train))])

# b)
print("Reshaping...")
x_train = reshapeToVector(x_train)
x_val = reshapeToVector(x_val)
x_test = reshapeToVector(x_test)
print("Reshaping: DONE\n")

print("Rescaling... ")
rescale(x_train)
rescale(x_val)
rescale(x_test)
print("\nRescaling: DONE")


# PART B

# c)
def sigmoid(x):
    """
    Sigmoid function
    :param x: a numpy array to apply sigmoid function
    :return: the result of applying sigmoid
    """
    return 1.0 / (1.0 + np.exp(-x))


def max_likelihood(y_true, y_pred):
    """
    Calculates maximum likelihood
    :param y_true: a numpy array of true values
    :param y_pred: a numpy array of predicted values
    :return: the maximum likelihood
    """
    return np.dot(y_true.T, np.log(y_pred)) + np.dot((1 - y_true).T, np.log(1 - y_pred))


def logistic_regression(x, y, max_iterations=50, optimizer=0):
    """Trains logistic regression model using gradient ascent
    to gain maximum likelihood on the training data
    Args:
        x : a numpy array of data
        y : a numpy array of label
        max_iterations : number of max iterations
        optimizer : the optimizer for gradient ascent
    Returns: array with likelihoods
    """
    likelihoods = []
    theta = np.zeros(x.shape[1])

    # Perform gradient ascent
    for _ in tqdm.tqdm(range(max_iterations)):
        # Output probability value by applying sigmoid on z
        h = sigmoid(np.dot(x, theta))
        # Calculate the gradient values
        gradient = np.mean((y - h) * x.T, axis=1)
        # Update the weights
        theta = theta + optimizer * gradient
        # Calculating max likelihood
        likelihood = max_likelihood(y, h)
        likelihoods.append(likelihood)

    return likelihoods


# Evolution of the cost by iteration
for opt in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    a = logistic_regression(x_train, y_train, 100, opt)
    plt.plot(list(range(len(a))), a)
    plt.title('Evolution of the cost by iteration for optimizer: ' + str(opt))
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()
