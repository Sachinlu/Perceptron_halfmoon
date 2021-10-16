import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import perceptron as neural

# Loading Dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt", sep="\t")

# Data Normalization
df.columns = ['B', 'G', 'R', 'y']
X1 = df[df['y'] == 1]
X2 = df[df['y'] == 2]
X1 = X1.sample(n=1000)
X2 = X2.sample(n=1000)
df = X1.append(X2)
X = df[['B', 'G', 'R']].to_numpy()
y = df['y'].to_numpy()
y = np.where(y == 2, 0, y)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Training the Model
single_perceptron = neural.P_model()
single_perceptron.fit(X_train, y_train)
predictions = single_perceptron.prediction(X_test)

# Evaluation of the model
from sklearn.metrics import accuracy_score
print('This Program applies the single layer perceptron model for classification')
print("The dataset used:- Skin Segmentation Dataset from UCI- Machine learning Repository")
prediction_on_training_data = single_perceptron.prediction(X_train)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)
print(f'Accuracy on training data : {accuracy_on_training_data} ')

prediction_on_test_data = single_perceptron.prediction(X_test)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)
print(f'Accuracy on test data : {accuracy_on_test_data}')

# Plotting the graph
df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
colors = {0: "green", 1: "grey"}
fig, ax = plt.subplots()
grouped = df.groupby("label")
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

a1 = np.amin(X_train[:, 0])
a2 = np.amax(X_train[:, 0])

b1 = (-single_perceptron.weights[0] * a1 - single_perceptron.Bias) / single_perceptron.weights[1]
b2 = (-single_perceptron.weights[0] * a2 - single_perceptron.Bias) / single_perceptron.weights[1]


ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin - 3, ymax + 3])

plt.show()
