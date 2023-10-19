import numpy as np
import scipy.special
import progressbar

class NeuralNetwork:
    def __init__(
        self, numInputNodes, numOutputNodes, learningRate, *numHiddenNodesPerLayer
    ):
        self.numInputNodes = numInputNodes + 1
        self.numHiddenNodesPerLayer = [
            numHiddenNodes + 1 for numHiddenNodes in numHiddenNodesPerLayer
        ]
        self.numOutputNodes = numOutputNodes
        self.numLayers = len(numHiddenNodesPerLayer) + 1

        self.learningRate = learningRate

        self.activation = lambda x: scipy.special.expit(x)

        self.weights = []

        if self.numLayers <= 1 or self.numHiddenNodesPerLayer[0] <= 1:
            self.weights.append(
                np.random.normal(
                    0.0,
                    pow(self.numOutputNodes, -0.5),
                    (self.numOutputNodes, self.numInputNodes),
                )
            )
        else:
            self.weights.append(
                np.random.normal(
                    0.0,
                    pow(self.numHiddenNodesPerLayer[0] - 1, -0.5),
                    (self.numHiddenNodesPerLayer[0] - 1, self.numInputNodes),
                )
            )

            for i in range(1, self.numLayers - 1):
                self.weights.append(
                    np.random.normal(
                        0.0,
                        pow(self.numHiddenNodesPerLayer[i] - 1, -0.5),
                        (
                            self.numHiddenNodesPerLayer[i] - 1,
                            self.numHiddenNodesPerLayer[i - 1],
                        ),
                    )
                )

            self.weights.append(
                np.random.normal(
                    0.0,
                    pow(self.numOutputNodes, -0.5),
                    (self.numOutputNodes, self.numHiddenNodesPerLayer[-1]),
                )
            )

    def train(self, inputs, targets):
        inputs = np.append(np.array(inputs, ndmin=2), [[1]], axis=1).T
        targets = np.array(targets, ndmin=2).T

        activationInputs = [None]
        activationOutputs = [inputs]
        for layerIdx in range(self.numLayers):
            activationInputs.append(
                np.dot(self.weights[layerIdx], activationOutputs[-1])
            )
            activationOutputs.append(self.activation(activationInputs[-1]))
            if layerIdx < self.numLayers - 1:
                activationOutputs[-1] = np.append(activationOutputs[-1], [[1]], axis=0)
        # print("Activation Inputs:\n", activationInputs)
        # print("Activation Outputs:\n", activationOutputs)

        errors = self.calculateErrors(targets, activationOutputs)
        # print("Errors:\n", errors)
        pass
        self.updateWeights(activationOutputs, errors)
        # print("------------------------------------------")

    def calculateErrors(self, targets, activationOutputs):
        errors = [0 for layer in range(self.numLayers)]
        errors[-1] = targets - activationOutputs[-1]
        for layerIdx in range(1, self.numLayers):
            if layerIdx == 1:
                errors[-(layerIdx + 1)] = np.dot(
                    self.weights[-layerIdx].T, errors[-layerIdx]
                )
            else:
                errors[-(layerIdx + 1)] = np.dot(
                    self.weights[-layerIdx].T, errors[-layerIdx][:-1]
                )
        return errors

    def updateWeights(self, activationOutputs, errors):
        for layerIdx in range(1, self.numLayers + 1):
            # print(f"WEIGHTS:\n{self.weights}")
            layerError = errors[-layerIdx]
            activationOutput = activationOutputs[-layerIdx]
            prevActivationOutput = activationOutputs[-(layerIdx + 1)]
            # print(f"Layer Error:\t {layerError.shape}\n{layerError}")
            # print(f"Activation Output:\t {activationOutput.shape}\n{activationOutput}")
            # print(f"Prev Activation Output T:\t {prevActivationOutput.T.shape}\n{prevActivationOutput.T}")
            product = layerError * activationOutput * (1.0 - activationOutput)
            if layerIdx > 1:
                product = product[:-1]
            # print(f"Product:\t{product.shape}\n{product}")
            # print(f"Weight:\n{self.weights[-layerIdx]}")
            dotProduct = (prevActivationOutput @ product.T).T
            # print(f"dotProduct:\n{dotProduct}")
            self.weights[-layerIdx] += self.learningRate * dotProduct
            # print("==========================================")

    def query(self, inputs):
        inputs = np.append(np.array(inputs, ndmin=2), [[1]], axis=1).T

        activationInputs = [None]
        activationOutputs = [inputs]
        for layerIdx in range(self.numLayers):
            activationInputs.append(
                np.dot(self.weights[layerIdx], activationOutputs[-1])
            )
            activationOutputs.append(self.activation(activationInputs[-1]))
            if layerIdx < self.numLayers - 1:
                activationOutputs[-1] = np.append(activationOutputs[-1], [[1]], axis=0)

        return activationOutputs[-1]

    

        # for index in range(len(inputs)):
        #     print("Inputs:\t", inputs[index])
        #     print("Outputs:\t", outputs[index])
