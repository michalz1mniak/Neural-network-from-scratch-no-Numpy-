# importing data
X = []
with open('circles_X.csv') as f:
    for line in f:
        X.append(list(map(float,line.strip().split(','))))
y = []
with open('circles_y.csv') as f:
    for line in f:
        y.append(int(line.strip()))

# creating neurons
# n11 - first neuron in first layer
# w - weight/weights
# b - bias

import random

# hidden layer
n11w = [random.uniform(-1,1), random.uniform(-1,1)]
n12w = [random.uniform(-1,1), random.uniform(-1,1)]
n13w = [random.uniform(-1,1), random.uniform(-1,1)]

n11b = random.uniform(-1,1)
n12b = random.uniform(-1,1)
n13b = random.uniform(-1,1)

# second layer
n21w = [random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)]

n21b = random.uniform(-1,1)

# ReLU function
def relu(x):
    return max(0,x)

# sigmoid function
import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# cost function
def cost_f(outn, realout):
    return (outn-realout)**2

epochs = 10000
learning_rate = 0.003

for epoch in range(1, epochs + 1):
    total_error = 0  
    
    for i in range(200):
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

        # calculating error
        error = cost_f(z21, y_real)
        total_error += error  

        delta_out = (z21 - y_real) * z21 * (1 - z21)

        grad_n21w0 = delta_out * z11
        grad_n21w1 = delta_out * z12
        grad_n21w2 = delta_out * z13
        grad_n21b = delta_out

        n21w[0] -= learning_rate * grad_n21w0
        n21w[1] -= learning_rate * grad_n21w1
        n21w[2] -= learning_rate * grad_n21w2
        n21b -= learning_rate * grad_n21b

        delta_h1 = delta_out * n21w[0] * (z11 > 0)
        delta_h2 = delta_out * n21w[1] * (z12 > 0)
        delta_h3 = delta_out * n21w[2] * (z13 > 0)

        grad_n11w0 = delta_h1 * x1
        grad_n11w1 = delta_h1 * x2
        grad_n11b = delta_h1

        grad_n12w0 = delta_h2 * x1
        grad_n12w1 = delta_h2 * x2
        grad_n12b = delta_h2

        grad_n13w0 = delta_h3 * x1
        grad_n13w1 = delta_h3 * x2
        grad_n13b = delta_h3

        n11w[0] -= learning_rate * grad_n11w0
        n11w[1] -= learning_rate * grad_n11w1
        n11b -= learning_rate * grad_n11b

        n12w[0] -= learning_rate * grad_n12w0
        n12w[1] -= learning_rate * grad_n12w1
        n12b -= learning_rate * grad_n12b

        n13w[0] -= learning_rate * grad_n13w0
        n13w[1] -= learning_rate * grad_n13w1
        n13b -= learning_rate * grad_n13b
   
    print(f"Epoch: {epoch}, average error: {total_error / 200:.4f}")


with open('parameters.txt', 'w') as f:
    f.write(f'{n11w[0]} {n11w[1]}\n')
    f.write(f'{n12w[0]} {n12w[1]}\n')
    f.write(f'{n13w[0]} {n13w[1]}\n')
    f.write(f'{n11b}\n')
    f.write(f'{n12b}\n')
    f.write(f'{n13b}\n')
    f.write(f'{n21w[0]} {n21w[1]} {n21w[2]}\n')
    f.write(f'{n21b}\n')
    
    