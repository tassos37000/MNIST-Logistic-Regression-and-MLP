import tensorflow as tf
import numpy as np
# for visuals
import tqdm
import matplotlib.pyplot as plt


# PART A
def filterClasses(x, y):
    """
    Keeps only classes 5 and 6, we assume class 5 is 0 and class 6 is 1
    :param x: dataset features
    :param y: dataset labels
    :return: the dataset only with filtered classes
    """
    keep = (y == 5) | (y == 6)
    x, y = x[keep], y[keep]
    y = y == 6
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

x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0
print("Rescaling: DONE")


# # PART B
#
# # c)
def sigmoid(x):
    """
    Sigmoid function
    :param x: a numpy array to apply sigmoid function
    :return: the result of applying sigmoid
    """
    return 1.0 / (1.0 + np.exp(-x))


def cost_log_reg(y_true, y_hat, _lambda=0.0, theta=0):
    """
    Calculates the cost for logistic regression
    :param y_true: a numpy array of true values
    :param y_hat: a numpy array of predicted values
    :param _lambda: lambda for regularisation
    :param theta: a numpy array of weights for regularisation
    :return: the cost
    """
    return np.dot(y_true.T, np.log(y_hat)) + np.dot((1 - y_true).T, np.log(1 - y_hat)) - (_lambda / 2.0) * np.sum(theta ** 2)


def fitLogisticRegression(x, y, max_iterations=50, optimizer=0.0, _lambda=0.0):
    """
    Trains logistic regression model using gradient ascent to gain maximum likelihood on the training data
    :param x: a numpy array of data
    :param y: a numpy array of label
    :param max_iterations: number of max iterations (default is 50)
    :param optimizer: the optimizer for gradient ascent (default is 0.0)
    :param _lambda: lambda for regularisation (default is 0.0)
    :returns: array with costs and numpy array of final weights
    """
    # Array with costs
    J_train = []
    # Initialize weights
    theta = np.zeros(x.shape[1])

    # Perform gradient ascent
    for _ in range(max_iterations):
        h = sigmoid(np.dot(x, theta))
        # Calculating cost
        cur_cost = cost_log_reg(y, h, _lambda, theta)
        # regularization e)
        reg = _lambda * theta
        # Calculate the gradient values
        gradient = np.mean((y - h) * x.T, axis=1) - reg
        # Update the weights
        theta = theta + optimizer * gradient
        # Save the cost
        J_train.append(cur_cost)

    return J_train, theta


# d)
LR_train, LR_theta = fitLogisticRegression(x_train, y_train, 100, 0.1)
# LR_train.pop(0)
# plt.plot(np.arange(len(LR_train)), LR_train)
# plt.title('Evolution of the cost by iteration for optimizer: ' + str(0.3))
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.show()


def predictLogisticRegression(theta, x):
    """
    Predicts labels of a data set given a weight table
    :param theta: a numpy array with weights
    :param x: a numpy array with data
    :return: a numpy array with 0,1 if data belongs in that label
    """
    p = sigmoid(np.dot(x, theta))
    p = p > 0.5 - 1e-6
    return p


p_test = predictLogisticRegression(LR_theta, x_test)
print('Accuracy of test set', round(np.mean(p_test.astype('int') == y_test) * 100, 3), "%")

# e)
print("\ne)")
list_l2 = np.linspace(10**-4, 10, num=100)
theta_l2 = []
accuracy_l2 = []

for i in tqdm.tqdm(range(len(list_l2))):
    _, LR_theta = fitLogisticRegression(x_val, y_val, 100, 0.1, list_l2[i])
    theta_l2.append(LR_theta)
    p_test = predictLogisticRegression(LR_theta, x_val)
    accuracy_l2.append(np.mean(p_test.astype('int') == y_val) * 100)

# Show accuracy by L2
plt.plot(list_l2, accuracy_l2)
plt.xlabel('L2')
plt.ylabel('accuracy')
plt.show()

# accuracy of best model with L2 regularization
min_val_error = np.where(accuracy_l2 == max(accuracy_l2))
p_test = predictLogisticRegression(theta_l2[min_val_error[0][0]], x_test)
print('Accuracy of test set with L2 regularization', round(np.mean(p_test.astype('int') == y_test) * 100, 3), "%")

# PART C
#
# st)
y_train = y_train.reshape(y_train.shape[0], 1)
y_val = y_val.reshape(y_val.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)


def compute_cost(yhat, y):
    """
    Computes the cross-entropy cost
    :param yhat: numpy array of forward propagation
    :param y: numpy array of true values
    :returns: cost
    """
    cost = -np.sum(np.multiply(y, np.log(yhat)) + np.multiply(1 - y, np.log(1 - yhat))) / y.shape[0]
    cost = float(np.squeeze(cost))  # np.squeeze returns cost without arrays
    return cost


def softmax(x, ax=1):
    m = np.max(x, axis=ax, keepdims=True)  # max per row
    p = np.exp(x - m)
    return (p / np.sum(p, axis=ax, keepdims=True))


def cost_grad_sigmoid(W, X, t, lamda):
    # X: NxD
    # W: KxD
    # t: NxD

    E = 0
    N, D = X.shape
    K = t.shape[1]

    y = softmax(np.dot(X, W))

    for n in range(N):
        for k in range(K):
            E += t[n][k] * np.log(y[n][k])
    E -= lamda * np.sum(np.square(W)) / 2

    gradEw = np.dot((t - y).T, X) - lamda * W.T

    return E, gradEw


def gradcheck_sigmoid(Winit, X, t, lamda):
    W = np.random.rand(*Winit.shape)
    epsilon = 1e-6

    _list = np.random.randint(X.shape[0], size=5)
    x_sample = np.array(X[_list, :])
    t_sample = np.array(t[_list, :])

    Ew, gradEw = cost_grad_sigmoid(W, x_sample, t_sample, lamda)

    print("gradEw shape: ", gradEw.shape)

    numericalGrad = np.zeros(gradEw.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad
    for k in tqdm.tqdm(range(numericalGrad.shape[0])):
        for d in range(numericalGrad.shape[1]):
            # add epsilon to the w[k,d]
            w_tmp = np.copy(W)
            w_tmp[d, k] += epsilon
            e_plus, _ = cost_grad_sigmoid(w_tmp, x_sample, t_sample, lamda)

            # subtract epsilon to the w[k,d]
            w_tmp = np.copy(W)
            w_tmp[d, k] -= epsilon
            e_minus, _ = cost_grad_sigmoid(w_tmp, x_sample, t_sample, lamda)

            # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)

    return gradEw, numericalGrad


def fit_MLP(X, y, X_valid=np.array([]), y_valid=np.array([]), n_hidden=100, n_iterations=0, learning_rate=0.1):
    """
    MLP algorithm with as many input neurons as features, one hidden layer with n_hidden neurons and
    one neuron for output layer. Activation function for hidden and output layer is the Sigmoid function.
    With gradient descent and Early stopping based on the validation set.
    :param X: a numpy array with the data
    :param y: a numpy array with the labels
    :param X_valid: a numpy array with the validation data
    :param y_valid: a numpy array with the validation labels
    :param n_hidden: number of neurons for hidden layer
    :param n_iterations: number of iterations
    :param learning_rate: learning rate for gradient descent
    :return: a tuple with first table of weights(w1), first vector opf biases(b1),
                          second table of weights(w2), second vector opf biases(b2),
                          cost calculated on validating data, number of epochs
    """
    # Number of neurons in each layer
    n0, n1, n2 = (X.shape[1], n_hidden, 1)
    np.random.seed(3)

    # Initialize weights and biases
    w1 = np.random.randn(n0, n1) * 0.01  # * 0.01 because data are [0,1]
    b1 = np.zeros((n1,))
    w2 = np.random.randn(n1, n2) * 0.01  # * 0.01 because data are [0,1]
    b2 = np.zeros((n2,))

    count_error = 0
    epoch = 0
    cost = 100000
    # for _ in tqdm.tqdm(range(n_iterations)):  # uncomment this line to test st)
    while True:  # comment this line to test st)
        # Forward Pass
        z1 = np.dot(X, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        yhat = sigmoid(z2)
        # cur_cost = round(compute_cost(yhat, y), 2)  # uncomment this line to test st)
        # ..............

        # Early stopping # comment all early stopping to test st)
        z1_val = np.dot(X_valid, w1) + b1
        f1_val = sigmoid(z1_val)
        z2_val = np.dot(f1_val, w2) + b2
        yhat_val = sigmoid(z2_val)
        cur_cost = round(compute_cost(yhat_val, y_valid), 3)

        if cur_cost >= cost:
            count_error += 1
        else:
            count_error = 0

        if count_error == 5:
            print("Number of epoch from early stopping: ", epoch)
            return w1, b1, w2, b2, cost, epoch
        cost = cur_cost
        # ..............

        # Backward Pass
        dz2 = yhat - y
        dz1 = np.dot(dz2, w2.T) * (1 - a1) * a1
        dw1 = np.dot(X.T, dz1) / n0
        db1 = 1 / n0 * np.sum(dz1, axis=0, keepdims=True)
        dw2 = np.dot(a1.T, dz2) / n0
        db2 = 1 / n0 * np.sum(dz2, axis=0, keepdims=True)
        # ...............

        # update weights and biases
        w1 = w1 - learning_rate * dw1
        b1 = b1 - learning_rate * db1
        w2 = w2 - learning_rate * dw2
        b2 = b2 - learning_rate * db2
        # ...........

        # # Gradient checking
        # gradEw, numericalGrad = gradcheck_sigmoid(w1, X, y, learning_rate)
        # # Absolute norm
        # print("The difference estimate for gradient of w1 is : ", np.max(np.abs(gradEw - numericalGrad)))
        # gradEw, numericalGrad = gradcheck_sigmoid(w2, a1, y, learning_rate)
        # # Absolute norm
        # print("The difference estimate for gradient of w2 is : ", np.max(np.abs(gradEw - numericalGrad)))
        # # ...........

        # print progress
        predictions = np.select([yhat < 0.5, yhat >= 0.5], [0, 1])
        if epoch % 200 == 0:
            print("epoch", epoch, "cost:-->", cur_cost, "accuracy-->", float(np.squeeze(sum(y == predictions) / len(y) * 100)))
        epoch += 1

    # return w1, b2, w2, b2, cost, epoch  # uncomment this line to test st)


def predict_MLP(X, w0, b0, w1, b1):
    """
    Predicts the output layer given the weights nd biases
    :param X: a numpy array with the data
    :param w0: a numpy array with the weights from input to hidden layer
    :param b0: a numpy array with the biases from input to hidden layer
    :param w1: a numpy array with the weights from hidden to output layer
    :param b1: a numpy array with the biases from hidden to output layer
    :return: a numpy array with the computed output
    """
    Z1 = X.dot(w0) + b0
    A1 = sigmoid(Z1)
    Z2 = A1.dot(w1) + b1
    pred = sigmoid(Z2)
    return pred


# Test st)
# a = fit_MLP(x_train, y_train, 100, 500, 0.1)
# y_pred_train = predict_MLP(x_train, a[0], a[1], a[2], a[3])
# predictions_train = np.select([y_pred_train < 0.5, y_pred_train >= 0.5], [0, 1])
# print('Accuracy of train set', round(float(np.squeeze(sum(y_train == predictions_train) / len(y_train) * 100)), 3))


# h)
learn_rate = np.linspace(10**-5, 0.5, num=10)
M = [2**i for i in range(1, 11)]
n_epochs = []
best_model = (0, 0, (0, 0, 0, 0, 10000))  # (number of neurons for hidden, learning rate, model)
for m in M:
    n_epochs_for_m = []
    for lr in learn_rate:
        print("\nLearning rate: ", lr, " Number of hidden layers: ", m)
        a = fit_MLP(x_train, y_train, x_val, y_val, m, 1000, lr)
        if a[4] < best_model[2][4]:
            best_model = (m, lr, a)
        n_epochs_for_m.append(a[5])
    n_epochs.append(n_epochs_for_m)

print("Best model parameters:\n\tM: ", best_model[0], "\n\th: ", best_model[1], "\n\tE: ", best_model[2][5])

# Show each plot separately
# for i in range(len(M)):
#     plt.plot(learn_rate, n_epochs[i])
#     plt.title('Evolution of the Ε by η for M: ' + str(M[i]))
#     plt.xlabel('η')
#     plt.ylabel('E')
#     plt.show()

# All plots together
for i in range(len(M)):
    plt.plot(learn_rate, n_epochs[i], label='M = ' + str(M[i]))

plt.title('Evolution of the Ε by η')
plt.legend()
plt.xlabel('η')
plt.ylabel('E')
plt.show()


# theta)
y_pred_test = predict_MLP(x_test, best_model[2][0], best_model[2][1], best_model[2][2], best_model[2][3])
predictions_test = np.select([y_pred_test < 0.5, y_pred_test >= 0.5], [0, 1])
print('Accuracy of test set', round(float(np.squeeze(sum(y_test == predictions_test) / len(y_test) * 100)), 3))
