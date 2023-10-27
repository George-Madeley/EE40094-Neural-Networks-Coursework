from neuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import progressbar
import csv

sourceFileName = "./fashion_MNIST/fashion_mnist_"
destinationFileName = "./results_fashion_2.csv"

df_test_data = pd.read_csv(f"{sourceFileName}test.csv", header=None)
test_labels = np.array(df_test_data[0])
test_data = (np.asfarray(df_test_data[range(1, df_test_data.shape[1])]) / 255 * 0.99) + 0.01
numTestRows, numTestCols = test_data.shape

df_train_data = pd.read_csv(f"{sourceFileName}train.csv", header=None)
train_labels = np.array(df_train_data[0])
train_data = (
    (np.asfarray(df_train_data[range(1, df_train_data.shape[1])]) / 255 * 0.99) + 0.01
)
numTrainRows, numTrainCols = train_data.shape

numUniqueOutputs = len(df_train_data[0].unique())

print("train data")
y_value = np.zeros((1, numUniqueOutputs))
for i in range(numUniqueOutputs):
    print("occurance of ", i, "=", np.count_nonzero(train_labels == i))
    y_value[0, i - 1] = np.count_nonzero(train_labels == i)

# converting train_label in one hot encoder representation
train_labels_OHE = np.zeros((numTrainRows, numUniqueOutputs)) + 0.01
for rowIdx in range(numTrainRows):
    label = train_labels[rowIdx]
    for colIdx in range(numUniqueOutputs):
        if label == colIdx:
            train_labels_OHE[rowIdx, label] = 0.99
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

NUM_INPUT_NODES = numTrainCols
NUM_OUTPUT_NODES = numUniqueOutputs

learningRates = [base / 10 ** exponent for exponent in range(1, 3) for base in range(9, 0, -1)]

def getWidgets(message):
    return [
        f" {message}: [",
        progressbar.Percentage(),
        "]",
        progressbar.Bar(),
        " (",
        progressbar.Timer(),
        ") ",
    ]

fields = ["No. Hidden Nodes"] + [str(learningRate) for learningRate in learningRates]
with open(destinationFileName, "w", newline='') as outputCSV:
    writer = csv.DictWriter(outputCSV, fieldnames=fields)
    writer.writeheader()

for numHiddenNodes in range(900, 1000, 50):
    results = {"No. Hidden Nodes": numHiddenNodes}
    for learningRateIdx in progressbar.progressbar(
        range(len(learningRates)), redirect_stdout=True, widgets=getWidgets(f"Hidden Nodes: {numHiddenNodes}")
        ):
        learningRate = learningRates[learningRateIdx]
        neuralNetwork = NeuralNetwork(
            NUM_INPUT_NODES, NUM_OUTPUT_NODES, learningRate, numHiddenNodes
        )

        outputsNN = np.zeros(test_labels_OHE.shape, dtype=float)

        for rowIdx in range(numTrainRows):
            row = train_data[rowIdx]
            targets = train_labels_OHE[rowIdx]
            neuralNetwork.train(row, targets)

        for rowIdx in range(numTestRows):
            row = test_data[rowIdx]
            output = neuralNetwork.query(row)
            outputNN = outputsNN[rowIdx]
            outputsNN[rowIdx] = np.reshape(output, (numUniqueOutputs))

        maximumValues = np.amax(outputsNN, axis=1)
        maximumValues = np.tile(maximumValues, (numUniqueOutputs, 1))
        outputsNN = np.floor(outputsNN / maximumValues.T)
        outputsNN = outputsNN * test_labels_OHE
        numPassed = outputsNN.sum()

        print(f"Hidden Nodes:\t{numHiddenNodes}\tLearning Rate:\t{learningRate}\tPassed:\t{(numPassed/numTestRows)*100}%")
        results[str(learningRate)] = numPassed
    
    with open(destinationFileName, "a", newline='') as outputCSV:
        writer = csv.DictWriter(outputCSV, fieldnames=fields)
        writer.writerow(results)

