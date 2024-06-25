import numpy as np

np.random.seed(42) 
X = 2 * np.random.rand(100, 1) 
y = 4 + 3 * X + np.random.randn(100, 1)

# Hyperparameters
learning_rate = 0.01
n_iterations = 1000
m = len(X)

# Initialize parameters
w = np.random.randn(1) ## taking a random w_0
b = np.random.randn(1) ## random bias

# Gradient Descent
for iteration in range(n_iterations):
    # Compute predictions
    y_pred = X.dot(w) + b
    
    # Compute the loss (Mean Squared Error)
    loss = (1/m) * np.sum((y_pred - y)**2)
    
    # Compute the gradients
    gradient_w = (2/m) * X.T.dot(y_pred - y)
    gradient_b = (2/m) * np.sum(y_pred - y)
    
    # Update the parameters
    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b
    
    # Print the loss every 100 iterations
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: Loss = {loss}")

print(f"Final parameters: w = {w[0]}, b = {b[0]}")

# Plotting the results
import matplotlib.pyplot as plt

plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, X.dot(w) + b, color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
