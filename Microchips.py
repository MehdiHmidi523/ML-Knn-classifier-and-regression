import numpy as np
import matplotlib.pylab as plt
from collections import Counter
from matplotlib.colors import ListedColormap

"""
Load dataset from `A1_datasets` into the variables X and y.
X should be a numpy array of n lines and 2 columns (the input data matrix).
y should be a numpy array of n elements (the outputs vector).
"""
filename = "A1_datasets/microchips.csv"
data = np.genfromtxt(filename, delimiter=",")
X = data[:, :2]
y = data[:, -1]

""" 
Produce a scatter plot of the training dataset
To see that the dataset cannot be separated into positive (class 1) and negative (class 0) examples 
by a straight-line through the plot. 
"""
X0 = X[y == 0]
X1 = X[y == 1]

fig, ax = plt.subplots()
ax.scatter(X0[:, 0], X0[:, 1], color='red', label='Rejected (y=0)', marker='x')
ax.scatter(X1[:, 0], X1[:, 1], color='blue', label='Accepted (y=1)', marker='+')
ax.set_ylabel('Microship Test 2')
ax.set_xlabel('Microship Test 1')
plt.legend()
fig.show()

"""
Compute the euclidean distance between two arrays u and v.
"""


def dist(u, v):
    return np.linalg.norm(u - v)


""" 
Return the predicted class-label for x, using the training dataset X, y, and k nearest neighbours.
"""


def prediction(x, X, y, k=3):
    listOfDistances = np.array([dist(x, row) for row in X])
    class_labels = y[np.argsort(listOfDistances)][:k]
    return Counter(class_labels).most_common(1)[0][0]


"""
function prediction(x, X, y, k = in [1, 3, 5, 7] ) on x = chip1, chip2, chip3; 
returns the class-label 1 (i.e. accepted microShip).
"""
for i in [1, 3, 5, 7]:
    print("chip1, chip2, chip3 for ---- k = {}".format(i))
    x = np.array([-0.3, 1.0])
    print(prediction(x, X, y, k=i))
    x = np.array([-0.5, -0.1])
    print(prediction(x, X, y, k=i))
    x = np.array([0.6, 0.0])
    print(prediction(x, X, y, k=i))

""" 
plots the decision boundary and the training dataset
"""


def plot_decision_boundary(func, X, y, k):
    print("Plotting... for k = {}".format(k))
    min_x1, max_x1 = min(X[:, 0]) - 0.1, max(X[:, 0]) + 0.1
    min_x2, max_x2 = min(X[:, 1]) - 0.1, max(X[:, 1]) + 0.1

    plot_x1, plot_x2 = np.meshgrid(np.linspace(min_x1, max_x1, 50),
                                   np.linspace(min_x2, max_x2, 50))
    points = np.c_[plot_x1.ravel(), plot_x2.ravel()]
    preds = np.array([func(x, X, y, k) for x in points])
    preds = preds.reshape(plot_x1.shape)

    X0 = X[y == 0]
    X1 = X[y == 1]

    fig, ax = plt.subplots()
    ax.pcolormesh(plot_x1, plot_x2, preds, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    ax.scatter(X0[:, 0], X0[:, 1], color="red", label="Rejected")
    ax.scatter(X1[:, 0], X1[:, 1], color="blue", label="Accepted")
    ax.set_xlabel("Microship Test 1")
    ax.set_xlabel("Microship Test 2")
    ax.set_title("Decision boundary with k = {}".format(k))
    plt.legend()
    fig.show()


"""
Plot the decision boundary 
depending on K we see the difference in the decision boundary. It is more complex when k is smaller.
"""
plot_decision_boundary(prediction, X, y, k=1)
plot_decision_boundary(prediction, X, y, k=3)
plot_decision_boundary(prediction, X, y, k=5)
plot_decision_boundary(prediction, X, y, k=7)

""" 
Compute the the training error by comparing the predicted class-labels 
with the original class-labels y. for k = 1,3,5,7.
"""
predictedClasses = []

for row, datapoint in enumerate(X):
    copiedArray = np.array((np.copy(X)).tolist())
    rowToClassify = copiedArray[row]
    copiedArray = np.array((np.delete(copiedArray, row, 0)).tolist())
    predictedClasses.append(prediction(rowToClassify, copiedArray, y, k=5))


correctClasses = 0
for predictedClass, originalClass in zip(predictedClasses, y):
    if predictedClass == originalClass:
        correctClasses += 1

print(f'Training Error: {100 - (correctClasses / len(predictedClasses) * 100)}%')
