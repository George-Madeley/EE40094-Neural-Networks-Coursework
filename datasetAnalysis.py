import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math

fileName = "./fashion_MNIST/fashion_mnist_test.csv"

df = pd.read_csv(fileName, header=None)
labels = np.array(df[0])
numRows, numCols = df.shape

numUniqueOutputs = len(df[0].unique())

numOccurances = []
for i in range(numUniqueOutputs):
    numOccurances.append(np.count_nonzero(labels == i))

df_occurances = pd.DataFrame(list(zip(range(numUniqueOutputs), numOccurances)), columns=["label", "occurances"])
df_occurances.occurances /= numRows
df_occurances.occurances *= 100

ax = sns.barplot(x="label", y="occurances", data=df_occurances)
ax.set(xlabel="Label", ylabel="Occurances (%)")
for index, row in df_occurances.iterrows():
    ax.text(row.label, row.occurances, round(row.occurances, 2), color='black', ha="center")
plt.show()