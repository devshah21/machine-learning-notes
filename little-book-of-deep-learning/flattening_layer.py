import numpy as np

# Example input tensor with shape (2, 2, 3)
input_tensor = np.array([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
])

# Flatten the tensor
flattened_vector = input_tensor.flatten()

print("Input Tensor Shape:", input_tensor.shape)
print("Input Tensor:\n", input_tensor)
print("Flattened Vector Shape:", flattened_vector.shape)
print("Flattened Vector:\n", flattened_vector)

