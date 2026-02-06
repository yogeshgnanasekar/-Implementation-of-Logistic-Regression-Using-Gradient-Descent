# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Initialize the model parameters (weights and bias) with small values and set the learning rate.

Compute the predicted output using the sigmoid function for the given input features.

Calculate the loss (error) and update the weights and bias using gradient descent.

Repeat the update process until convergence, then use the trained model for prediction.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Yogesh G
RegisterNumber:  25009804
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("C:/Users/91908/Downloads/Placement_Data (1).csv")

data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

X = data[['ssc_p', 'mba_p']].values
y = data['status'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

m = len(y)
X = np.c_[np.ones(m), X]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))




theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []

for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)

y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)

accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")

plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show()


```

## Output:
<img width="452" height="215" alt="Screenshot 2026-02-06 113854" src="https://github.com/user-attachments/assets/d8e471a3-54e4-4b46-9d02-95891520d661" />
<img width="598" height="439" alt="Screenshot 2026-02-06 113903" src="https://github.com/user-attachments/assets/620474ce-ca97-4862-8fbd-f71fa722f903" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

