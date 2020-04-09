import struct
import numpy as np
from collections import Counter


def dist(u, v):
    return np.linalg.norm(u - v)


def prediction(x, X, y, k=3):
    listOfDistances = np.array([dist(x, row) for row in X])
    class_labels = y[np.argsort(listOfDistances)][:k]
    return Counter(class_labels).most_common(1)[0][0]


# Step 1. Data Preparation
# Read functions credits to github user tylerneylon
# https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


# Training data setup
raw_train = read_idx("A1_datasets/train-images-idx3-ubyte.gz")
train_data = np.reshape(raw_train(60000, 28 * 28)) # trouble with reshape
train_label = read_idx("A1_datasets/train-labels-idx1-ubyte.gz")

# Test data setup
raw_test = read_idx("A1_datasets/t10k-images-idx3-ubyte.gz")
test_data = np.reshape(raw_test, (10000, 28 * 28))
test_label = read_idx("A1_datasets/t10k-labels-idx1-ubyte.gz")

# Step 2. Knn prediction fit for similar looking digits 5, 3 and 8
idx = (train_label == 5) | (train_label == 3) | (train_label == 8)
X = train_data[idx]
y = train_label[idx]

idx = (test_label == 5) | (test_label == 3) | (test_label == 8)
x_test = test_data[idx]
y_true = test_label[idx]


for i in x_test:
    y_predict = prediction(i, X, y, k=5)

#print(int(y_predict))
