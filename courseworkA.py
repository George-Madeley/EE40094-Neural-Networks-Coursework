import numpy as np
import scipy.special

class NeuralNetwork:
    def __init__(self, numInputNodes, numOutputNodes, learningRate, *numHiddenNodes):
        self.numInputNodes = numInputNodes
        self.numHiddenNodesPerLayer = numHiddenNodes
        self.numOutputNodes = numOutputNodes
        self.numHiddenLayers = len(numHiddenNodes)

        self.learningRate = learningRate

        self.activation = lambda x: scipy.special.expit(x)

        self.weights = []
        self.weights.append(np.random.normal(
            0.0,
            pow(self.numHiddenNodesPerLayer[0], -0.5),
            (self.numHiddenNodesPerLayer[0], self.numInputNodes)
        ))
        for i in range(1, self.numHiddenLayers):
            self.weights.append(np.random.normal(
                0.0,
                pow(self.numHiddenNodesPerLayer[i], -0.5),
                (self.numHiddenNodesPerLayer[i], self.numHiddenNodesPerLayer[i - 1])
            ))
        self.weights.append(np.random.normal(
            0.0,
            pow(self.numOutputNodes, -0.5),
            (self.numOutputNodes, self.numHiddenNodesPerLayer[-1])
        ))

    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        activationInputs = [None]
        activationOutputs = [inputs]
        for layerIdx in range(self.numHiddenLayers + 1):
            activationInputs.append(np.dot(self.weights[layerIdx], activationOutputs[-1]))
            activationOutputs.append(self.activation(activationInputs[-1]))

        errors = self.calculateErrors(targets, activationOutputs)
        self.updateWeights(activationOutputs, errors)

    def calculateErrors(self, targets, activationOutputs):
        errors = [0 for layer in range(self.numHiddenLayers + 1)]
        errors[-1] = targets - activationOutputs[-1]
        for layerIdx in range(2, self.numHiddenLayers + 2):
            errors[-layerIdx] = np.dot(self.weights[-(layerIdx - 1)].T, errors[-(layerIdx - 1)])
        return errors

    def updateWeights(self, activationOutputs, errors):
        for weightIdx in range(self.numHiddenLayers + 1):
            layerError = errors[-weightIdx]
            activationOutput = activationOutputs[-weightIdx]
            prevActivationOutput = activationOutputs[-(weightIdx + 1)]
            self.weights[weightIdx] += self.learningRate * np.dot(
                (layerError * activationOutput * (1.0 - activationOutput)),
                np.transpose(prevActivationOutput)
            )

    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2).T

        activationInputs = [None]
        activationOutputs = [inputs]
        for layerIdx in range(self.numHiddenLayers + 1):
            activationInputs.append(np.dot(self.weights[layerIdx], activationOutputs[-1]))
            activationOutputs.append(self.activation(activationInputs[-1]))

        return activationOutputs[-1][0][0]
    
neuralNetwork = NeuralNetwork(2, 1, 0.1, 2)
inputs = [
  [0,0],
  [0,1],
  [1,0],
  [1,1]
]
targets = [0, 0, 0, 1]
for index in range(len(inputs)):
    neuralNetwork.train(inputs[index], targets[index])

for index in range(len(inputs)):
    output = neuralNetwork.query(inputs[index])
    print("Inputs: ", inputs[index])
    print("Output: ", output)