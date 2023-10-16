import numpy as np
import scipy.special

class NeuralNetwork:
    def __init__(self, numInputNodes, numOutputNodes, learningRate, *numHiddenNodesPerLayer):
        self.numInputNodes = numInputNodes + 1
        self.numHiddenNodesPerLayer = [numHiddenNodes + 1 for numHiddenNodes in numHiddenNodesPerLayer]
        self.numOutputNodes = numOutputNodes
        self.numLayers = len(numHiddenNodesPerLayer) + 1

        self.learningRate = learningRate

        self.activation = lambda x: scipy.special.expit(x)

        
        self.weights = []

        if self.numLayers <= 1 or self.numHiddenNodesPerLayer[0] <= 1:
            self.weights.append(np.random.normal(
                0.0,
                pow(self.numOutputNodes, -0.5),
                (self.numOutputNodes, self.numInputNodes)
            ))
        else:
            self.weights.append(np.random.normal(
                0.0,
                pow(self.numHiddenNodesPerLayer[0] - 1, -0.5),
                (self.numHiddenNodesPerLayer[0] - 1, self.numInputNodes)
            ))

            for i in range(1, self.numLayers - 1):
                self.weights.append(np.random.normal(
                    0.0,
                    pow(self.numHiddenNodesPerLayer[i] - 1, -0.5),
                    (self.numHiddenNodesPerLayer[i] - 1, self.numHiddenNodesPerLayer[i - 1])
                ))

            self.weights.append(np.random.normal(
                0.0,
                pow(self.numOutputNodes, -0.5),
                (self.numOutputNodes, self.numHiddenNodesPerLayer[-1])
            ))

    def train(self, inputs, targets):
        inputs = np.append(np.array(inputs, ndmin=2), [[1]], axis=1).T
        targets = np.array(targets, ndmin=2).T

        activationInputs = [None]
        activationOutputs = [inputs]
        for layerIdx in range(self.numLayers):
            activationInputs.append(np.dot(self.weights[layerIdx], activationOutputs[-1]))
            activationOutputs.append(self.activation(activationInputs[-1]))
            if layerIdx < self.numLayers - 1:
                activationOutputs[-1] = np.append(activationOutputs[-1], [1])

        errors = self.calculateErrors(targets, activationOutputs)
        self.updateWeights(activationOutputs, errors)

    def calculateErrors(self, targets, activationOutputs):
        errors = [0 for layer in range(self.numLayers)]
        errors[-1] = targets - activationOutputs[-1]
        for layerIdx in range(1, self.numLayers):
            errors[-(layerIdx + 1)] = np.dot(self.weights[-layerIdx].T, errors[layerIdx])
        return errors

    def updateWeights(self, activationOutputs, errors):
        for weightIdx in range(1, self.numLayers + 1):
            layerError = errors[-weightIdx]
            activationOutput = activationOutputs[-weightIdx]
            prevActivationOutput = activationOutputs[-(weightIdx + 1)]
            self.weights[weightIdx - 1] += self.learningRate * np.dot(
                (layerError * activationOutput * (1.0 - activationOutput)),
                np.transpose(prevActivationOutput)
            )

    def query(self, inputs):
        inputs = np.append(np.array(inputs, ndmin=2), [[1]], axis=1).T

        activationInputs = [None]
        activationOutputs = [inputs]
        for layerIdx in range(self.numLayers):
            activationInputs.append(np.dot(self.weights[layerIdx], activationOutputs[-1]))
            activationOutputs.append(self.activation(activationInputs[-1]))
            if layerIdx < self.numLayers - 1:
                activationOutputs[-1] = np.append(activationOutputs[-1], [1])

        return activationOutputs[-1][0][0]
    
    def trainUntilPass(self, inputs, targets, maxIterations=10000, minPrecision=0.01):
        numIterations = 0
        precision = np.inf

        outputs = np.array([0 for target in targets], dtype=float)
        targets = np.array(targets)

        while numIterations < maxIterations and precision > minPrecision:
            for index in range(len(inputs)):
                self.train(inputs[index], targets[index])

            for index in range(len(inputs)):
                output = neuralNetwork.query(inputs[index])
                outputs[index] = output

            precision = np.sum(np.abs(targets - outputs)) / len(inputs)
            numIterations += 1

            print("Iteration:\t", numIterations)
            print("Precision:\t", precision)
            print("\n")

        for index in range(len(inputs)):
            print("Inputs:\t", inputs[index])
            print("Outputs:\t", outputs[index])

        

    
neuralNetwork = NeuralNetwork(2, 1, 0.1)
inputs = [
  [0,0],
  [0,1],
  [1,0],
  [1,1]
]
targets = [0, 0, 0, 1]
neuralNetwork.trainUntilPass(inputs, targets, maxIterations=100000, minPrecision=0.1)