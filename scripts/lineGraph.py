import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

df = pd.read_csv("./results/fashion_epochs.csv")

df["0.1"] = df["0.1"].apply(lambda x: x / 100)
df["0.01"] = df["0.01"].apply(lambda x: x / 100)

print(df["0.1"].max())
print(df["0.01"].max())

# A line graph with Epochs on the X-axis and performance of 0.1 and 0.01 on the Y-axis
sns.lineplot(
  data=df,
  x="Epochs",
  y="0.1",
  label="0.1",
)
sns.lineplot(
  data=df,
  x="Epochs",
  y="0.01",
  label="0.01",
)
plt.xlabel("Epochs")
plt.ylabel("Performance")
plt.title("Performance of Fashion MNIST with different epochs")
plt.show()
