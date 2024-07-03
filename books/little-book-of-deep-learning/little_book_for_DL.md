# little book for deep learning.

## Chapter 1

- to create relations between an input x and output y — collect large training set D of pairs (x_n, y_n) and devise a parametric model f
    - this is a piece of compute code that incorporates trainable parameters *w* that modulate its behaviour
        - such that, with the proper values *w*,* it is a good predictor
- “good” predictors is basically if an x is given to the code, the y’ value it computes is a good estimate of the y that would have been associated with x in the training set
    - when it is a good prediction, the loss is small
    - training the model consists of computing a value *w** which minimizes the los
- the trainable parameters that compose *w* are called weights
- models usually depend on hyper-parameters — which are set according to domain prior knowledge, best practices, or resource constraints
    - they can be optimized — but are different from the way the weights are optimized
- 3 categories can be used to describe ML models
    - regression (supervised)
    - classification (supervised)
    - density modelling — model the probability density function of the data itself (unsupervised)

## Chapter 2

- tensors are series of scalars arranged along several discrete axes
    - they are used to represent signal to be processed, trainable parameters of the models, and the intermediate quantities they compute (this last one is called activations)
- tensors are really good for computational efficiency

## Chapter 3

- cross entropy is defined as: $L = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$
    - the intuition is N is the number of samples
    - y_i is the true / correct label
    - p_i is the prediction
- **For *yi*=1:**
    - If the model predicts *pi*=1, the loss contribution from this sample is −log(1)=0.
    - If the model predicts *pi*=0, the loss contribution from this sample is −log(0), which is very large (approaches infinity).
- **For *yi*=0:**
    - If the model predicts *pi*=0, the loss contribution from this sample is −log(1−0)=0.
    - If the model predicts *pi*=1, the loss contribution from this sample is −log(1−1), which is very large (approaches infinity).
- now expanding this to multi-class, we get: $L = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \log(p_{ic})$
    - everything remains the same, but C is the number of classes
- chain rule for probability (basically multiplying a sequence of probabilities
    - this is super helpful when we have a sequence of tokens and we want to predict the next one → this is known as autoregressive generative model
- to train an autoregressive generative model what we want to do is minimize the sun across training sequences and time steps of the **cross-entropy loss**
- casual models are when we have a computational structure such that the compute logits l_t for x_t only depend on everything from x_1 to x_(t-1)
    - it’s called casual because it doesn’t let the future impact the past
- the conversion from words to token representation is carried out by an algo called the **tokenizer**
    - common one used is the byte-pair-encoding
- **gradient descent**
    - the optimal parameters *w** don’t have closed forms for something that isn’t linear regression
    - in the general case — to minimize a function, the go to is to use gradient descent
        - initialize a parameter with random value $w_0$ and then improve this estimate by iterating every gradient step
            - each step involves computing the gradient of the loss function with respect to the parameters and subtracting a fraction of it
                - $w_{n+1} = w_n - \eta \nabla \mathcal{L} \big|_w (w_n)$
                    - that weird symbol we’re multiplying by, that is called the learning rate
    - this entire gradient descent business is pretty much moving the current estimate in the direction that locally **decreases** the loss
- **learning rate**
    - positive value — modulates how quickly we minimize
        - if too small, learns super slow, if too large, the value will bounce around a good minimum and never descend into it
- **stochastic gradient descent**
    - take the full set *D* and split it into batches and update the parameters from the estimate of the gradient computed for each batch
        - this is much quicker since we’re updating after each training example / mini-batch
- **backpropagation**
    - the best way to understand backpropagation is via an example.
        - let’s take the MNIST dataset, let’s say the correct output for the image is the number 2, the model looks at the probability distribution across the 10 classes and figures out which predictions need to increase and which needs to decrease
            - once this is configured, we take 1 step back to the 2nd last layer (the last hidden layer) and implement these changes and understand which weights and biases need to be changed and by how much
                - we apply this process recursively for every layer before this
        - refer to this coding example:
    
    ```python
    def backward_pass(X, y, z1, a1, z2, a2):
        # Compute the error at the output
        # error_output is the difference between the predicted output and the actual output
        error_output = a2 - y
        
        # Compute delta at the output layer
        # delta_output is the error_output multiplied by the derivative of the sigmoid function at a2
        delta_output = error_output * sigmoid_derivative(a2)
        
        # Compute the error at the hidden layer
        # error_hidden is the dot product of delta_output and the transpose of W2
        error_hidden = np.dot(delta_output, W2.T)
        
        # Compute delta at the hidden layer
        # delta_hidden is the error_hidden multiplied by the derivative of the sigmoid function at a1
        delta_hidden = error_hidden * sigmoid_derivative(a1)
        
        # Compute gradients for W2 and b2
        # dW2 is the dot product of the transpose of a1 and delta_output
        # db2 is the sum of delta_output along the rows
        dW2 = np.dot(a1.T, delta_output)
        db2 = np.sum(delta_output, axis=0)
        
        # Compute gradients for W1 and b1
        # dW1 is the dot product of the transpose of X and delta_hidden
        # db1 is the sum of delta_hidden along the rows
        dW1 = np.dot(X.T, delta_hidden)
        db1 = np.sum(delta_hidden, axis=0)
        
        return dW1, db1, dW2, db2
    
    ```
    

## Chapter 4

<aside>
💡 very important chapter — focuses on understanding the many components of a deep learning model

</aside>

- types of layers
    - fully connected layers / dense
    - convolutional layers
    - pooling layer
    - dropout layer
- **fully connected layer**
    - each neuron has an associated weight  (stored in a matrix) and bias (stored as a vector)
        - weights determine the strength of the connection between neurons & bias helps in adjusting the output along with the weighted sum
            - to understand the above better — open this toggle
                
                Imagine you have a simple fully connected layer with two input neurons and one output neuron.
                
                1. **Inputs**: x1 and x2
                2. **Weights**: w1 and w2 (corresponding to the connections from x1 to the output neuron and x2 to the output neuron, respectively)
                3. **Bias**: b (associated with the output neuron)
                    
                    
                
                ### Step-by-Step Calculation
                
                1. **Input Multiplication by Weights**:
                    - Each input value is multiplied by its corresponding weight.
                    - If x1=0.5 and w1=0.8, then 0.5×0.8=0.4.
                    - If x2=0.3 and w2=0.2, then 0.3×0.2=0.06.
                2. **Weighted Sum**:
                    - The results from the multiplications are summed up.
                    - 0.4+0.06=0.46
                3. **Adding the Bias**:
                    - The bias b is added to the weighted sum.
                    - If b=0.1, then 0.46+0.1=0.56.
                4. **Activation Function**:
                    - The resulting value (0.56 in this case) is passed through an activation function to produce the final output.
                    - If using the ReLU activation function: ReLU(0.56)=0.56 (since ReLU outputs the input directly if it is positive).
                
    - each neuron computes a weighted sum of its inputs. if W is the weight matrix and b is the bias vector, the weighted sum for a single neuron is computed as: $z_i​=∑​W_{ij}​x_j​+b_i​$
        - W_ij is the weight connecting the j-th input neuron to the i-th neuron in the current layer, x_j is the j-th input and b_i is the bias for the i-th neuron
    - the weighted sum is then passed through an activation function (ReLU, Sigmoid, Tanh)
    - example calculation:
        
        Consider a simple example with a fully connected layer having 2 input neurons and 3 output neurons.
        
        - **Input Vector**: x=[x1,x2]
        - **Weight Matrix**: W of shape 3×2 (3 output neurons, 2 input neurons
        
        $$
        \mathbf{W} = \begin{bmatrix}
        W_{11} & W_{12} \\
        W_{21} & W_{22} \\
        W_{31} & W_{32}
        \end{bmatrix}
        $$
        
        - **Bias Vector**: b=[b1,b2,b3]
        - **Weighted Sum**: For each output neuron i,
        
        $$
        z_i = W_{i1} x_1 + W_{i2} x_2 + b_i  \text{ :general form}\\
        z_1 = W_{11} x_1 + W_{12} x_2 + b_1 \\
        z_2 = W_{21} x_1 + W_{22} x_2 + b_2 \\
        z_3 = W_{31} x_1 + W_{32} x_2 + b_3
        $$
        
        - **Activation Function**: Applying an activation function (e.g., ReLU) to each zi,
            - a1=ReLU(z1)
            - a2=ReLU(z2)
            - a3=ReLU(z3)
        
        The output of this fully connected layer is the vector a=[a1,a2,a3], which is then passed to the next layer in the neural network.
        
- **convolutional layer**
    - filters / kernels
        - these are small matrices that slide over the input data (a simple square matrix) → this is done to detect features such as edges, textures, or patterns
        - each filter is applied across the entire input data to produce a feature map
    - feature maps
        - the output of the filters is called a feature map
        - each feature map highlights the presence of certain features detected by the corresponding filter
    - stride
        - this is basically the number of pixels we move by
            - for example, a stride of 1 means we move 1 pixel at a time
    - padding
        - we add extra pixels (0 valued pixels) around teh data to control the spatial dimensions of the feature maps
            - there are many types of padding — we usually use “same” (padding to ensure output size == input size) or “valid” (no padding)
    - activation function
        - we apply activation functions — applied element-wise to the feature maps to introduce non-linearity
    - **how it works**
        - first we do the filter application. the filter slides over the input data. for each position, an element-wise multiplication is performed between the filter and the corresponding patch of the input data
        - the results of the multiplication are summed up to produce a singel value in the feature map
- **pooling layers (max or average)**
    - the goal of pooling layers are to help reduce the spatial dimensions of the input volume — this essentially helps decrease the number of parameters and computations in the network
        - this help prevent overfitting and make the neural network more computationally efficient
    - max pooling (similar process to average pooling)
        - we define a small window size (2x2 or 3x3) and strides (1, 2, etc) and then extract the max value per stride
            
            ```python
            1  3  2  4
            5  6  7  8
            9  1  2  0
            3  4  5  6
            
            max pooling with window size 2x2 and stride 2
            
            Window 1: [1  3; 5  6] -> Max: 6
            Window 2: [2  4; 7  8] -> Max: 8
            Window 3: [9  1; 3  4] -> Max: 9
            Window 4: [2  0; 5  6] -> Max: 6
            ```
            
    - other info about pooling layers
        - dimensionality reduction: pooling layers reduce spatial dimensions (height and width), but keeps the depth (# of channels) uncahnged
        - translation invariance: pooling helps the network become invariant to small translations of the input — meaning the network’s recognition capability becomes more robust to slight shifts and distortions
        - downsampling: by reducing the spatial dimensions — layers help in summarizing the presence of features in patches of the feature map — focusing on the most important aspects detected by the previous layers
- **normalization layers**
    - there are many types of normalization layers: batch, layer, instance, group
    - **batch normalization**
        - this type of normalization normalizes the input features across the min-batch — ensuring each feature has a mean of 0 and variance of 1
        - a min-batch of data with shape (N,D), where N is the batch size and D is the number of features
        - each feature, we compute the mean and variance across the mini-batch
        - after normalization, apply a scale and shift parameter to each feature:
            
            
            $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} , \quad y_i = \gamma \hat{x}_i + \beta$
            
- **dropout layer**
    - this is a regularization technique used to prevent overfitting
    - key concepts:
        - randomly dropping units
            - during each forward pass in training, a specified fraction of neurons in the dropout layer are randomly set to 0 — this basically means these neurons don’t contribute to the forward pass and are effectively ignored during the backpropagation step
        - dropout rate
            - straight froward, this is the probability of setting a neuron to 0
        - scaling
            - during training, the output of the non-dropped neurons are scaled by $\frac{1}{1-p}$ to ensure that the expected sum of the inpputs remains the same
- **flattening layer**
    - primary function is to transform multi-dimensional input data into a 1d vector
        - this transformation is important when we’re transitioning from convolutional layers (multi-dimensional data) to fully connected layers (1d vectors)
    - this layer doesn’t do anything else, it just acts as a bridge between the layers
- **activation functions**
    - the role of activation function is to introduce non-linearity — this basically allows it to learn from errors and accurately model complex data
        - it allows the model to approximate more complex functions — making it capable of solving a wider variety of problems
    - they also help map input values to a known range, which helps stabilize training and helps map values to a desired output in the last layer
    - there are many different types of activation functions
        - **sigmoid: $\frac{1}{1 + e^{-x}}$**
        - **tanh(x): $\frac{e^x - e^{-x}}{e^x + e^{-x}}$**
        - **relu:** *max(0, x)*
        - **softmax: $\frac{e^{x_i}}{\sum{e^{x_j}}}$**
- **putting everything together**
    - so when training our model, what we do is first apply the linear transformation or convolution operation or whatever. then we pass the input to the activation function and then we would pass it to a pooling layer.
        - this is for the forward pass
    - for the backward pass, we first compute the loss, then we compute the gradient of the loss with respect to the output activation
        - compute the gradient of the loss with respect to the weights and biases of the output layer → then update weights and biases of the output layer
            - repeat this process for the hidden layers

## Chapter 5

- **understanding MLPs (multi-layer perceptrons)**
    - this basically takes the form of multiple FCC layer (described above) separated by  activation functions
        - most common activation function used for these are ReLU
    - here’s what the training process looks like:
        - **a)** Initialize weights randomly
        - **b)** Perform forward propagation with a batch of training data
        - **c)** Compute the loss (error) between the predicted and actual outputs
        - **d)** Perform backpropagation to compute gradients
        - **e)** Update weights using an optimization algorithm (e.g., Stochastic Gradient Descent)
        - **f)** Repeat steps b-e for multiple epochs until convergence

- **understanding residual networks**
    - this is an architecture used for the famous CNN called ResNet — it’s used to solve the issue for vanishing gradients (using residual connections)
        - these residual connections allow for 100s of layers
            - A residual connection allows the input to a layer to be directly added to the output of a later layer. This creates a "shortcut" for information flow through the network.
                - In a standard neural network layer: y = F(x) With a residual connection: y = F(x) + x. Where x is the input, F(x) is the layer's transformation, and y is the output