import numpy as np
import scipy.special
import matplotlib.pyplot as plt

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
                activationOutputs[-1] = np.append(activationOutputs[-1], [[1]], axis=0)
        # print("Activation Inputs:\n", activationInputs)
        # print("Activation Outputs:\n", activationOutputs)

        errors = self.calculateErrors(targets, activationOutputs)
        # print("Errors:\n", errors)
        self.updateWeights(activationOutputs, errors)
        # print("------------------------------------------")

    def calculateErrors(self, targets, activationOutputs):
        errors = [0 for layer in range(self.numLayers)]
        errors[-1] = targets - activationOutputs[-1]
        for layerIdx in range(1, self.numLayers):
            errors[-(layerIdx + 1)] = np.dot(self.weights[-layerIdx].T, errors[layerIdx])
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
            product = (layerError * activationOutput * (1.0 - activationOutput))
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
            activationInput = np.dot(self.weights[layerIdx], activationOutputs[-1])
            if len(activationInput.shape) != 2:
                activationInput = np.reshape(activationInput, (activationInput.shape[0], 1))
            activationInputs.append(activationInput)
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

    def showStateSpaceRepresentation(self, inputs, targets):
        for layerIdx in range(self.numLayers):
            if layerIdx == 0:
                if self.numInputNodes == 3:
                    bias = self.weights[0][0][-1]
                    weights = self.weights[0][0][:-1]
                    print(bias)
                    print(weights)
                    seperatorLineX = np.linspace(-1, 2, 100)
                    seperatorLineY = (-weights[0]/weights[1])*seperatorLineX-(bias/weights[1])
                    plt.plot(seperatorLineX, seperatorLineY)

        for inputPair, target in zip(inputs, targets):
            color = "red" if target == 0 else "green"
            plt.scatter(inputPair[0], inputPair[1], s=50, color=color, zorder=3)

        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)

        plt.xlabel("Input A")
        plt.ylabel("Input B")
        plt.title("State Space of Input Vector")

        plt.grid(True, linewidth=1, linestyle=":")
        plt.tight_layout()
        plt.show()

        

    
neuralNetwork = NeuralNetwork(2, 1, 0.1, 2)
inputs = [
  [0,0],
  [0,1],
  [1,0],
  [1,1]
]
targets = [0, 1, 1, 1]
neuralNetwork.trainUntilPass(inputs, targets, maxIterations=10000, minPrecision=0.1)
neuralNetwork.showStateSpaceRepresentation(inputs, targets)