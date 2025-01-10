with open('parameters.txt') as f:
    parameters = f.readlines()

# hidden layer
n11w = list(map(float,parameters[0].split()))
n12w = list(map(float,parameters[1].split()))
n13w = list(map(float,parameters[2].split()))

n11b = float(parameters[3].strip())
n12b = float(parameters[4].strip())
n13b = float(parameters[5].strip())

# second layer
n21w = list(map(float,parameters[6].split()))

n21b = float(parameters[7].strip())


X = []
with open('circles_X_test.csv') as f:
    for line in f:
        X.append(list(map(float,line.strip().split(','))))
y = []
with open('circles_y_test.csv') as f:
    for line in f:
        y.append(int(line.strip()))

# ReLU function
def relu(x):
    return max(0,x)

# sigmoid function
import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

pred = []
for i in range(50):
        x1 = X[i][0]
        x2 = X[i][1]
        y_real = y[i]

        a11 = x1 * n11w[0] + x2 * n11w[1] + n11b
        a12 = x1 * n12w[0] + x2 * n12w[1] + n12b
        a13 = x1 * n13w[0] + x2 * n13w[1] + n13b

        z11 = relu(a11)
        z12 = relu(a12)
        z13 = relu(a13)

        a21 = z11 * n21w[0] + z12 * n21w[1] + z13 * n21w[2] + n21b
        z21 = sigmoid(a21)
        pred.append(round(z21))


# plotting
import matplotlib.pyplot as plt

X1 = [para[0] for para in X]
X2 = [para[1] for para in X]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X1, X2, c = y, cmap='bwr', marker='o', s=100, alpha=0.7)
plt.title("Original data")
plt.xlabel("X1")
plt.ylabel("X2")

plt.subplot(1, 2, 2)
plt.scatter(X1, X2, c = pred, cmap='bwr', marker='o', s=100, alpha=0.7)
plt.title("Predictions")
plt.xlabel("X1")
plt.ylabel("X2")

plt.show()