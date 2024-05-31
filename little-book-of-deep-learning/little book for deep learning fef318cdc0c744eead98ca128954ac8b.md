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
    -