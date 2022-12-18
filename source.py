# PART A
import tensorflow as tf
import numpy as np
# for visuals
import tqdm
# import matplotlib.pyplot as plt


def filterClasses(x, y):
    """
    Keeps only classes 5 and 6, we assume class 5 is 1 and class 6 is 0
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
    Reshape images from 28x28 to a 784 vector
    :param train: dataset to reshape
    :return: a numpy array with the reshapes dataset and with type float64
    """
    resh_train = []
    for j in range(len(train)):
        resh_train.append(np.reshape(np.array(train[j]), 784).astype("float64"))
    return np.array(resh_train)


def rescale(train):
    """
    Rescales dataset [0,255] -> [0,1]
    :param train: dataset
    :return: the rescaled dataset
    """
    for k in tqdm.tqdm(range(len(train))):
        for j in range(len(train[k])):
            if train[k][j] == 0:
                continue
            train[k][j] = train[k][j] / 255.0


# a)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, y_train = filterClasses(x_train, y_train)
x_test, y_test = filterClasses(x_test, y_test)
print("Loading: DONE")

# create validation set from train set
(x_val, y_val) = (x_train[int(0.8 * len(x_train)):], y_train[int(0.8 * len(y_train)):])
(x_train, y_train) = (x_train[:int(0.8 * len(x_train))], y_train[:int(0.8 * len(y_train))])

# b)
x_train = reshapeToVector(x_train)
x_val = reshapeToVector(x_val)
x_test = reshapeToVector(x_test)
print("Reshaping: DONE")

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


def cost_log_reg(y_true, y_pred, _lambda, theta):
    """
    Calculates the cost for logistic regression
    :param y_true: a numpy array of true values
    :param y_pred: a numpy array of predicted values
    :param _lambda: lambda for regularisation
    :param theta: a numpy array of weights for regularisation
    :return: the cost
    """
    return np.dot(y_true.T, np.log(y_pred)) + np.dot((1 - y_true).T, np.log(1 - y_pred)) - (_lambda / 2.0) * np.sum(theta ** 2)


def logistic_regression(x, y, max_iterations=50, optimizer=0.0, _lambda=0.0):
    """
    Trains logistic regression model using gradient ascent to gain maximum likelihood on the training data
    :param x: a numpy array of data
    :param y: a numpy array of label
    :param max_iterations: number of max iterations (default is 50)
    :param optimizer: the optimizer for gradient ascent (default is 0.0)
    :param _lambda: lambda for regularisation (default is 0.0)
    :returns: array with costs and numpy array of final weights
    """
    J_train = []
    theta = np.zeros(x.shape[1])

    # Perform gradient ascent
    for _ in range(max_iterations):
        # Output probability value by applying sigmoid on z
        h = sigmoid(np.dot(x, theta))
        # Calculating max likelihood
        cur_cost = cost_log_reg(y, h, _lambda, theta)
        # regularization e)
        reg = _lambda * theta
        # Calculate the gradient values
        gradient = np.mean((y - h) * x.T, axis=1) - reg
        # Update the weights
        theta = theta + optimizer * gradient

        J_train.append(cur_cost)

    return J_train, theta


# d)
LR_train, LR_theta = logistic_regression(x_train, y_train, 100, 0.1)
# LR_train.pop(0)
# plt.plot(np.arange(len(LR_train)), LR_train)
# plt.title('Evolution of the cost by iteration for optimizer: ' + str(0.3))
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.show()


def predict(theta, x):
    p = sigmoid(np.dot(x, theta))
    prob = p
    p = p > 0.5 - 1e-6

    return p, prob


p_test, prob_test = predict(LR_theta, x_test)
print('Accuracy of test set', round(np.mean(p_test.astype('int') == y_test) * 100, 3), "%")

print("\nPART E\n")
list_l2 = np.linspace(10**-4, 10, num=100)
theta_l2 = []
accuracy_l2 = []

for i in tqdm.tqdm(range(len(list_l2))):
    _, LR_theta = logistic_regression(x_train, y_train, 100, 0.1, list_l2[i])
    theta_l2.append(LR_theta)
    p_test, prob_test = predict(LR_theta, x_val)
    accuracy_l2.append(np.mean(p_test.astype('int') == y_val) * 100)

min_val_error = np.where(accuracy_l2 == max(accuracy_l2))
p_test, prob_test = predict(theta_l2[min_val_error[0][0]], x_test)
print('Accuracy of test set', round(np.mean(p_test.astype('int') == y_test) * 100, 3), "%")
