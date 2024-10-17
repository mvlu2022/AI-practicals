import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

# Initialize parameters
input_size = 3    # Number of input features
hidden_size = 4   # Number of neurons in the hidden layer
output_size = 1   # Number of output neurons
learning_rate = 0.1
epochs = 10000

# Initialize weights and biases
np.random.seed(42)  # For reproducibility
W1 = np.random.randn(input_size, hidden_size) * 0.01
print("W1 is :",W1)
b1 = np.zeros((1, hidden_size))
print("b1 is: ",b1)
W2 = np.random.randn(hidden_size, output_size) * 0.01
print("w2 is:",W2)
b2 = np.zeros((1, output_size))
print("B2 is: ",b2)

# Training data
X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [1, 0, 0]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Backward pass
    output_error = y - a2
    output_delta = output_error * sigmoid_derivative(a2)

    hidden_error = output_delta.dot(W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(a1)

    # Update weights and biases
    W2 += a1.T.dot(output_delta) * learning_rate
    b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(hidden_delta) * learning_rate
    b1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.square(y - a2))
        print(f'Epoch {epoch}, Loss: {loss}')

# Final output after training
print("Final output after training:")
print(a2)
