import numpy as np
import matplotlib.pylab as plt

filename = "A1_datasets/polynomial200.csv"
data = np.genfromtxt(filename, delimiter=",")
# Divide the dataset into a training set of size 100, and test set of size 100.
X = data[:100, 0]
y = data[:100, 1]
X1 = data[100:, 0]
y1 = data[100:, 1]

# Plot the training and test set side-by-side in a 1 × 2 pattern.
fig, ax = plt.subplots()
ax.plot(X, y, 'ob', label='training_set', markersize=3)
ax.set_ylabel('Polynomial Test 2')
ax.set_xlabel('Polynomial Test 1')
plt.legend()
fig.show()

fig, ax = plt.subplots()
ax.plot(X1, y1, 'ro', label='test_set', markersize=3)
ax.set_ylabel('Polynomial Test 2')
ax.set_xlabel('Polynomial Test 1')
plt.legend()
fig.show()


# Predict a real number instead of a class. Instead of classifying a test instance based on the most frequently
# occurring class among the k nearest neighbors, we take the average of the target variable of the k nearest neighbors.
def dist(u, v):
    return np.linalg.norm(u - v)


def prediction(x, X, y, k=3):
    listOfDistances = np.array([dist(x, row) for row in X])
    class_labels = y[np.argsort(listOfDistances)][:k]
    return class_labels.mean()


for k in [1, 3, 5, 7]:
    # Calculate the prediction for every x in the training set and the test set
    training_prediction_array = np.array([])
    test_prediction_array = np.array([])
    for x in range(len(X)):
        training_prediction_array = np.append(training_prediction_array, prediction(X[x], X, y, k))
        test_prediction_array = np.append(test_prediction_array, prediction(X1[x], X, y, k))

    # Calculate MSE for the training and test data
    mse_training = np.mean((training_prediction_array[:] - y[:]) ** 2)
    mse_test = np.mean((test_prediction_array[:] - y1[:]) ** 2)

    xy = []
    for x in np.arange(0, 25, 0.25):
        t = prediction(x, X, y, k)
        xy.append([x, t])
    regression = np.asarray(xy)

    # training regression over plot
    fig, ax = plt.subplots()
    plt.title("polynomial_train, k=" + str(k) + ", MSE=" + str(round(mse_training, 2)))
    ax.plot(X, y, 'o', label='training_data', markersize=3)
    ax.plot(regression[:, 0], regression[:, 1], color='royalblue')
    ax.set_ylabel('Polynomial Test 2')
    ax.set_xlabel('Polynomial Test 1')
    plt.legend()
    fig.show()

    # MSE
    print("MSE test error:")
    print("k=" + str(k) + ", MSE=" + str(round(mse_test, 2)))

print("The best result is for k=5, MSE=28.48")