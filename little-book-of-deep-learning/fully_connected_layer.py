import numpy as np

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases randomly
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size)
        
    def forward(self, inputs):
        # Compute the weighted sum
        weighted_sum = np.dot(self.weights, inputs) + self.biases
        # Apply the ReLU activation function
        output = relu(weighted_sum)
        return output

# Example usage
if __name__ == "__main__":
    # Input vector (2 inputs)
    inputs = np.array([0.5, 0.3])

    # Create a fully connected layer with 2 inputs and 1 output
    fc_layer = FullyConnectedLayer(input_size=2, output_size=1)
    
    # Perform forward pass
    output = fc_layer.forward(inputs)
    
    # Print the output
    print("Output:", output)
    print("Weights:", fc_layer.weights)
    print("Biases:", fc_layer.biases)
