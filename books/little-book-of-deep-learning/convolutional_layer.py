import numpy as np

def convolve2d(image, kernel, stride=1, padding=0):
    # Add padding to the image
    image_padded = np.pad(image, padding, mode='constant', constant_values=0)
    kernel_height, kernel_width = kernel.shape
    output_height = (image_padded.shape[0] - kernel_height) // stride + 1
    output_width = (image_padded.shape[1] - kernel_width) // stride + 1
    
    # Initialize the output feature map
    output = np.zeros((output_height, output_width))
    
    # Perform the convolution operation
    for y in range(0, output_height):
        for x in range(0, output_width):
            region = image_padded[y*stride:y*stride+kernel_height, x*stride:x*stride+kernel_width]
            output[y, x] = np.sum(region * kernel)
    
    return output

# Example usage
if __name__ == "__main__":
    # Example image (5x5)
    image = np.array([
        [1, 2, 3, 0, 1],
        [0, 1, 2, 1, 0],
        [2, 1, 0, 1, 2],
        [1, 0, 1, 2, 1],
        [0, 1, 2, 1, 0]
    ])
    
    # Example filter (3x3)
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])
    
    # Convolve the image with the filter
    feature_map = convolve2d(image, kernel, stride=1, padding=0)
    
    # Print the feature map
    print("Feature Map:\n", feature_map)
