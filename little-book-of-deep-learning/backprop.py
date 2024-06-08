import numpy as np

# Define the structure of the network
input_size = 2
hidden_size = 2
output_size = 1

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_pass(X):
    z1 = np.dot(X, W1) + b1 ## y = mx + b
    a1 = sigmoid(z1) # apply the activation function
    z2 = np.dot(a1, W2) + b2 ## do y = mx + b again
    a2 = sigmoid(z2) # activation function (again)
    return z1, a1, z2, a2

def backward_pass(X, y, z1, a1, z2, a2):
    # Compute the error at the output
    error_output = a2 - y
    delta_output = error_output * sigmoid_derivative(a2)
    
    # Compute the error at the hidden layer
    error_hidden = np.dot(delta_output, W2.T)
    delta_hidden = error_hidden * sigmoid_derivative(a1)
    
    # Compute gradients
    dW2 = np.dot(a1.T, delta_output)
    db2 = np.sum(delta_output, axis=0)
    dW1 = np.dot(X.T, delta_hidden)
    db1 = np.sum(delta_hidden, axis=0)
    
    return dW1, db1, dW2, db2

def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate=0.1):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Define a simple dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR problem

# Train the network
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward pass
    z1, a1, z2, a2 = forward_pass(X)
    
    # Backward pass
    dW1, db1, dW2, db2 = backward_pass(X, y, z1, a1, z2, a2)
    
    # Update weights
    W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
    
    # Print the loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean((y - a2) ** 2)
        print(f"Epoch {epoch}, Loss: {loss}")

# Print final predictions
_, _, _, a2 = forward_pass(X)
print("Predictions after training:")
print(a2)
