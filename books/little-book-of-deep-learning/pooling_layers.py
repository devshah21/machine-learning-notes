import numpy as np

def max_pooling(input_matrix, size=2, stride=2):
    input_height, input_width = input_matrix.shape
    output_height = (input_height - size) // stride + 1
    output_width = (input_width - size) // stride + 1
    output = np.zeros((output_height, output_width))
    
    for y in range(0, output_height):
        for x in range(0, output_width):
            region = input_matrix[y*stride:y*stride+size, x*stride:x*stride+size]
            output[y, x] = np.max(region)
    
    return output

def average_pooling(input_matrix, size=2, stride=2):
    input_height, input_width = input_matrix.shape
    output_height = (input_height - size) // stride + 1
    output_width = (input_width - size) // stride + 1
    output = np.zeros((output_height, output_width))
    
    for y in range(0, output_height):
        for x in range(0, output_width):
            region = input_matrix[y*stride:y*stride+size, x*stride:x*stride+size]
            output[y, x] = np.mean(region)
    
    return output

# Example usage
if __name__ == "__main__":
    input_matrix = np.array([
        [1, 3, 2, 4],
        [5, 6, 7, 8],
        [9, 1, 2, 0],
        [3, 4, 5, 6]
    ])
    
    max_pooled = max_pooling(input_matrix)
    average_pooled = average_pooling(input_matrix)
    
    print("Max Pooled Output:\n", max_pooled)
    print("Average Pooled Output:\n", average_pooled)
