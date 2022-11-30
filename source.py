# PART A
import tensorflow as tf
import numpy as np
import tqdm


# keep classes 5 and 6
def filterClasses(x, y):
    keep = (y == 5) | (y == 6)
    x, y = x[keep], y[keep]
    y = y == 5
    return x, y


# reshape function
def reshapeToVector(train):
    resh_train = []
    for i in tqdm.tqdm(range(len(train))):
        resh_train.append(np.reshape(np.array(train[i]), 784).astype("float64"))
    return resh_train


# rescale to [0.0, 1.0]
def rescale(train):
    for i in tqdm.tqdm(range(len(train))):
        for j in range(len(train[i])):
            if train[i][j] == 0:
                continue
            # print("before train[i][j]: ", train[i][j])
            # print("train[i][j]/255.0: ", train[i][j]/255.0)
            # print(type(train[i][j]))
            train[i][j] = train[i][j]/255.0
            # print("after train[i][j]: ", train[i][j])


# a)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, y_train = filterClasses(x_train, y_train)
x_test, y_test = filterClasses(x_test, y_test)
print("Loading: DONE")

# validation set
(x_val, y_val) = (x_train[int(0.8*len(x_train)):], y_train[int(0.8*len(y_train)):])
(x_train, y_train) = (x_train[:int(0.8*len(x_train))], y_train[:int(0.8*len(y_train))])

# b)
print("Reshaping...")
x_train = reshapeToVector(x_train)
x_val = reshapeToVector(x_val)
x_test = reshapeToVector(x_test)
print("\nReshaping: DONE\n")

print("Rescaling... ")
rescale(x_train)
rescale(x_val)
rescale(x_test)
print("Rescaling: DONE")
