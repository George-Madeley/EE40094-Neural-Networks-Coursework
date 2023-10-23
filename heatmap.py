import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

df_old = pd.read_csv("./results_fashion.csv")
new_data = []

learningRates = [str(base / 10 ** exponent) for exponent in range(1, 3) for base in range(9, 0, -1)]
for learningRate in learningRates:
  df_old[learningRate] = df_old[learningRate].apply(lambda x: x / 100)
  values = df_old[learningRate]
  hiddenNodes = df_old["No. Hidden Nodes"]
  new_data += list(zip(
    hiddenNodes,
    [learningRate] * len(df_old[learningRate]),
    values
  ))

df_new = pd.DataFrame(
  data=new_data,
  columns=["No. Hidden Nodes", "learningRate", "Performance"]
)

minVal = math.ceil(df_new["Performance"].min())
maxVal = math.floor(df_new["Performance"].max())
center = df_new["Performance"].mean()
median = df_new["Performance"].median()

print("Min: " + str(minVal))
print("Max: " + str(maxVal))
print("Mean: " + str(center))
print("Median: " + str(median))

colors = sns.diverging_palette(0, 120, s=100, l=50, n=9, as_cmap=True)
colors = sns.blend_palette(["red", "yellow", "green"], as_cmap=True)
results = df_new.pivot(index="No. Hidden Nodes", columns="learningRate", values="Performance")

sns.heatmap(
  results,
  vmin=minVal + 5,
  vmax=maxVal - 1,
  cmap=colors,
  square=True,
)
plt.show()
