from neuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import progressbar

df_test_data = pd.read_csv("./MNIST/mnist_test.csv", header=None)
test_labels = np.array(df_test_data[0])
test_data = np.array(df_test_data[range(1, df_test_data.shape[1])], dtype=float) / 255
numTestRows, numTestCols = test_data.shape

df_train_data = pd.read_csv("./MNIST/mnist_train.csv", header=None)
train_labels = np.array(df_train_data[0])
train_data = (
    np.array(df_train_data[range(1, df_train_data.shape[1])], dtype=float) / 255
)
numTrainRows, numTrainCols = train_data.shape

numUniqueOutputs = len(df_train_data[0].unique())

print("train data")
y_value = np.zeros((1, 10))
for i in range(numUniqueOutputs):
    print("occurance of ", i, "=", np.count_nonzero(train_labels == i))
    y_value[0, i - 1] = np.count_nonzero(train_labels == i)

# converting train_label in one hot encoder representation
train_labels_OHE = np.zeros((numTrainRows, numUniqueOutputs))
for rowIdx in range(numTrainRows):
    label = train_labels[rowIdx]
    for colIdx in range(numUniqueOutputs):
        if label == colIdx:
            train_labels_OHE[rowIdx, label] = 1
print("train_data shape=" + str(np.shape(train_data)))
print("train_label shape=" + str(np.shape(train_labels_OHE)))

test_labels_OHE = np.zeros((numTestRows, numUniqueOutputs))
for rowIdx in range(numTestRows):
    label = test_labels[rowIdx]
    for colIdx in range(numUniqueOutputs):
        if label == colIdx:
            test_labels_OHE[rowIdx, label] = 1
print("test_data shape=" + str(np.shape(test_data)))
print("test_label shape=" + str(np.shape(test_labels_OHE)))

LEARNING_RATE = 0.3
NUM_INPUT_NODES = numTrainCols
NUM_OUTPUT_NODES = numUniqueOutputs

NUM_HIDDEN_NODES = [28]
print(f"Num Inputs:\t{NUM_INPUT_NODES}")
print(f"Num Hidden:\t{NUM_HIDDEN_NODES}")
print(f"Num Outputs:\t{NUM_OUTPUT_NODES}")
neuralNetwork = NeuralNetwork(
    NUM_INPUT_NODES, NUM_OUTPUT_NODES, LEARNING_RATE, *NUM_HIDDEN_NODES
)  

PASS_CRITERIA = 0.9

outputsNN = np.zeros(test_labels_OHE.shape, dtype=float)

widgets = [
    " TRAINING: [",
    progressbar.Percentage(),
    "]",
    progressbar.Bar(),
    " (",
    progressbar.Timer(),
    ") ",
]
for rowIdx in progressbar.progressbar(
    range(numTrainRows), redirect_stdout=True, widgets=widgets
):
    row = train_data[rowIdx]
    targets = train_labels_OHE[rowIdx]
    neuralNetwork.train(row, targets)

widgets = [
    " TESTING:  [",
    progressbar.Percentage(),
    "]",
    progressbar.Bar(),
    " (",
    progressbar.Timer(),
    ") ",
]
for rowIdx in progressbar.progressbar(
    range(numTestRows), redirect_stdout=True, widgets=widgets
):
    row = test_data[rowIdx]
    output = neuralNetwork.query(row)
    outputNN = outputsNN[rowIdx]
    outputsNN[rowIdx] = np.reshape(output, (numUniqueOutputs))

outputsNN = outputsNN * test_labels_OHE
numPassed = (outputsNN >= PASS_CRITERIA).sum()

print(f"Passed:\t{numPassed}/{numTestRows}")

# neuralNetwork.trainUntilPass(df_train, df_test, maxIterations=MAX_ITERATIONS, minPrecision=PRECISION)
# neuralNetwork.showStateSpaceRepresentation(inputs, targets)
