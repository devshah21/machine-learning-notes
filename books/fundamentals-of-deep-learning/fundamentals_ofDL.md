# Fundamentals of Deep Learning


## Chapter 1

- imagine we define a function `h(x, theta)` , let x be the input image and theta be a vector of the parameters that our program uses
    
    ![Screenshot 2023-07-11 at 2.17.54 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/edbcabc5-ec26-4592-ba89-c1f39d567c2e/Screenshot_2023-07-11_at_2.17.54_PM.png)
    
- This model is called a linear perceptron — been used since the 1950s
    - An optimal parameter vector theta positions the classifier so that we make as many correct predictions as possible
        - to do this, a technique called optimization is used → an optimizer aims to maximize the performance of a ML model by iteratively tweaking its parameters until the error is minimized
- The Neuron
    - our artificial neuron takes in some number of inputs x_1, x_2, etc
        - each input is multiplied by a weight and a constant bias is added to it
        - All of this is then passed through a function *f* to produce an output
- Feed-Forward Neural Networks
    - there are layers within the brain as well → the cerebral cortex is made up of 6 layers and information goes from one layer to another until sensory input is converted into conceptual understanding
        - for ex. the bottommost layer receives raw visual data from the eyes and then this information is processed by each layer and passed onto the next until in the last layer, we conclude whether we’re looking at a cat or soda can, etc
    - similarly for neural network, the bottom layer is the input data and the top layer computes out final answer

<aside>
🧩 Important things to note:

- hidden layers usually have fewer neurons than the input layer to force the network to learn compressed representations of the original input
- it’s not required that every neuron has its output connected to the inputs of all neurons in the next layer
- inputs + outputs are vectorized representations
</aside>

- Quick note:
    - linear neurons are usually easy to compute with, but they have limitations
        - any feed-forward network w/ linear neurons can be expressed as a network without hidden layers → this is bad because hidden layers enable us to learn important features from input data
- Sigmoid, Tanh, and ReLU Neurons
    - there are 3 major types of neurons that are used in practice that introduce nonlinearities in their computations
    - sigmoid:
        
        $$
        f(z) = \frac{1}{1 + e^{-z}}
        
        $$
        
        - this means that when the logit is very small, the output of a logistic neuron is very close to 0
        - when the logit is very large, the output of the logistic neuron is close to 1
    - Tanh
        - it’s a similar shaped nonlinearity, but it ranges from -1 to 1
    - ReLU - restricted linear unit
        - it uses the function: f(z) = max(0,z)

- Softmax Output Layers
    - most times, the output vector will be a probability distribution over a set of mutually exclusive labels
    - this is done by using a special output layer called a softmax layer
        - the output of a neuron in a softmax depends on the outputs of all the other neurons in its layer
            - this is because we require the sum of all the outputs to be equal to 1
    - a strong prediction would have a single entry in the vector close to 1, while the remaining entries close to 0

## Chapter 2; Training Feed-Forward Neural Nets.

- The main idea here is to understand how to figure out the parameter vectors
    - in simple terms, it’s accomplished by a process referred to as training
- One way to train the model to better recognize patterns and trends is to choose smart training examples, but this almost never helps in real situations
- If we want to calculate the output of the i_th training example, we want to train the neuron to pick the optimal weights
    - We want to minimize the value of the error function E
        
        $$
        E = \frac{1}{2} \sum_{i} (t^i - y^i)^2
        
        $$
        
        - To write this formally, if we know that t^i is the true answer and y^i is the value computed by the neural net, then we have the resulting equation above for squared error
- When E = 0, the model makes a perfect prediction on every training example

### Gradient Descent

- for ex if a linear neuron only has 2 inputs (weights: w_1 and w_2), then we can imagine a 3D space where the horizontal dimensions correspond to the weights and the vertical component is the error
    
    ![Screenshot 2023-07-12 at 4.05.50 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f6ba1d29-b9bd-46bb-b153-d4a6b0bc658a/Screenshot_2023-07-12_at_4.05.50_PM.png)
    
- we can also visualize the surface as a set of elliptical contours, where the min. error is the center of the ellipses
- contours correspond to setting of w_1 and w_2 that evaluate to the same val. of E
    - the closer the contours are to each other, the steeper the slope
    - the direction of the steepest descent is always perpendicular to the contours
        - this direction is expressed as a vector known as the gradient
- if we randomly initialize the weights, by evaluating the gradient at the current position, we can find the direction of the steepest descent, and then go in that direction
    - this process is repeated and eventually gets us to a point of min. error
        
        ![Screenshot 2023-07-12 at 4.18.23 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f74927aa-4897-4dae-a40d-659ce69d3302/Screenshot_2023-07-12_at_4.18.23_PM.png)
        
        - this is known as gradient descent

**Delta Rule + Learning Rates**

- In addition to the weight parameters, learning algorithms also need another parameter, which is called the `learning rate`
- Usually, at each step of moving perpendicular to the contour, we need to determine how far we want to walk before recalc. our new direction
    - the distance depends on the steepness of the surface → teh closer we are to the min. error, the less/shorter we want to step forward
    - However, if our error surface is rather mellow, training can take a lot of time, hence why we multiply the gradient by a factor epsilon, also known as the learning rate
- If the learning rate is too small → training takes too long
- If the learning rate is too big → we’ll diverge away from the minimum
- In order to calc how to change the weight, we eval. the gradient:
    
    ![Screenshot 2023-07-16 at 7.31.57 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3e191248-4873-49ae-8074-ab6dc699e66e/Screenshot_2023-07-16_at_7.31.57_PM.png)
    

### Backpropagation

- We don’t know what the hidden units are doing, but what we can do is compute how fast the error changes as we change a hidden activity
    - from there, we can figure out how fast the error changes when we change the weight of an individual connection
- we start by calculating the error derivatives with respect to a single training example
- each hidden unit can affect many output units, thus we combine many seperate effects on the error in an informative way
    - once we have the error derivatives for one layer, we use them to compute the error derivatives for the activities of the layer below

### Stochastic and Minibatch Gradient Descent

- the algorithm defined above is known as batch gradient descent
    - the idea behind that is using entire dataset to compute the error surface and then follow the gradient to take the path of steepest descent
        - however batch gradient descent is sensitive to saddle points → might lead to premature convergence
- the other approach is to use stochastic gradient descent (SGD), where at each iteration, our error surface is estimated only w/ respect to a single example
    - instead of a single static error surface, our error surface is dynamic
    - as a result, descending on this stochastic surface significantly improves our ability to navigate flat regions
        
        ![Screenshot 2023-07-17 at 2.35.55 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f5e99d3d-a1f4-4a5a-a6ab-18fc95b1cc0f/Screenshot_2023-07-17_at_2.35.55_PM.png)
        
- looking at error incurred one example at a time may not be a good enough approx. of the error surface → could make gradient descent take a lot of time
    - using mini-batch gradient descent is a potential fix
        - at every iteration, we compute the error surface with respect to some subset of the total dataset

### Overfitting + Validation data

- by building complex models to fit the data very well, it may perform poorly on training data → in other words, the model doesn’t generalize well
    - this is known as overfitting
- Preventing Overfitting:
    - Regularization: modifies the objective function that we minimize by adding additional terms that penalize large weights
        - In other words, we change the objective function so that it
        becomes *Error* + *λ f θ* , where *f θ* grows larger as the components of *θ* grow larger, and *λ* is the regularization strength
    - Max norm constraints
        - enforces an absolute upper bound on the magnitude of the incoming weight vector for every neuron and use projected gradient descent to enforce the constraint
    - Dropout
        - dropout is implemented by keeping a neuron active with some probability *p* or setting it to 0 otherwise
            - it prevents the network from becoming too dependent on any one of neurons
- At the end of each epoch, we want to measure how well our model is generalizing → this is known as validation set
    - at the end, it tells us how well it does on data that it hasn’t seen before

## Chapter 3; Implementing Neural Nets. in TF

- A variable is insufficient for passing input to model as it’s only initialized once → we need a component that we populate every time the computation grpah is run
    - we can use a placeholder
    
    ```python
    x = tf.placeholder(tf.float32, name = 'x', shape = [None, 784])
    W = tf.Variable(tf.random_uniform([784, 10], -1, 1), name = 'W')
    multiply = tf.matmul(x,W)
    ```
    
    - x represents a minibatch of data stored as float32s and has 784 columns, but an undefined number of rows
        - this means that x can be initialized with an arbitrary number of data samples

```python
import tensorflow as tf
from read_data import get_minibatch()

x = tf.placeholder(tf.float32, name="x", shape=[None, 784])
W = tf.Variable(tf.random_uniform([784, 10], -1, 1), name="W")
b = tf.Variable(tf.zeros([10]), name="biases")
output = tf.matmul(x, W) + b

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)
feed_dict = {"x" : get_minibatch()}
sess.run(output, feed_dict=feed_dict)
```

- first 4 lines after import describe the computational graph that is built by the session when it is finally instantiated
    - we then initialize the variables as required by using the session variable to run the initialization operation in `sess.run(init_op)`
    - we can run it again by calling sess.run, but now we pass a list of tensors we want to compute along with a feed_dict that fills the placeholders with the necessary input data
        - [sess.run](http://sess.run) basically traverses down the computational graph to identify all the dependencies that compose the relevant subgraphs
        
        ![Screenshot 2023-07-18 at 6.56.49 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e3c35cba-303c-45bd-b937-1cf241ac3ced/Screenshot_2023-07-18_at_6.56.49_PM.png)
        
- Key things
    
    ```python
    tf.get_variable(name, shape, initializer)
    # this checks if a varibale with this name exists and retrieves it if it does
    # otherwise it creates a new one with shape + init
    
    ```
    
- Managing models with CPU + GPU
    - /cpu:0
        - The CPU of the machine
    - /gpu:0
        - The first gpu of the machine, if it has one
    - /gpu:1
        - the second gpu of the machine, if it has one

### Logistic Regression Model in TF

- Method by which we calculate the probability that an input belongs in one of the target classes; example being used is MNIST
- Model uses a matrix W representing the weights of the connections in the network, as well as a vector b corresponding to the biases to estimate whether an input x belongs to class i using the softmax expression
    - our goal is to learn the values for W and b that most effectively classify our inputs as accurately as possible
        
        ![Screenshot 2023-07-19 at 9.26.35 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3f08b155-ab17-4a91-a148-af7ad30a3e1d/Screenshot_2023-07-19_at_9.26.35_PM.png)
        
- Output size of 10 because of 10 possible outcomes for each input
    - Input layer of 784 as input images are 28 x 28
- For building the model, there will be 4 phases
    - Inference; produces a probability distribution over the output classes given a minibatch
    - Loss; computes the value of the error function
    - Training; responsible for computing the gradients of the model’s parameters and updating the model
    - Evaluate; will determine the effectiveness of the model
- Given a minibatch which consists of 784 dimensional vectors representing MNIST images, we can represent logistic regression by taking the softmax of the input multiplied with a matrix representing the weights connecting the input and output layer
    
    ```python
    def inference(x):
             tf.constant_initializer(value=0)
             W = tf.get_variable("W", [784, 10],
                                 initializer=init)
             b = tf.get_variable("b", [10],
                                 initializer=init)
             output = tf.nn.softmax(tf.matmul(x, W) + b)
             return output
    ```
    
    ```python
    def loss(output, y):
            dot_product = y * tf.log(output)
            # Reduction along axis 0 collapses each column into a
            # single value, whereas reduction along axis 1 collapses
            # each row into a single value. In general, reduction along
            # axis i collapses the ith dimension of a tensor to size 1.
            xentropy = -tf.reduce_sum(dot_product, reduction_indices=1)
            loss = tf.reduce_mean(xentropy)
            return loss
    ```
    
    ```python
    def training(cost, global_step):
            optimizer = tf.train.GradientDescentOptimizer(
            learning_rate)
            train_op = optimizer.minimize(cost,
            global_step=global_step)
            return train_op
    ```
    
    ```python
    def evaluate(output, y):
             correct_prediction = tf.equal(tf.argmax(output, 1),
                                           tf.argmax(y, 1))
             accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                               tf.float32))
             return accuracy
    ```
    

### Logging & Training the Logistic Regression Model

- We can use `tf.scalar_summary` and `tf.histogram_summary` commands to log the cost for each minibatch, validation error, and the distribution of parameters
    
    ```python
    def training(cost, global_step):
             tf.scalar_summary("cost", cost)
             optimizer = tf.train.GradientDescentOptimizer(
            learning_rate)
             train_op = optimizer.minimize(cost,
            global_step=global_step)
             return train_op
    ```
    
    - Every epoch, we run `tf.merge_all_summaries` to collect all summary statistics we’ve logged
- Putting everything together, we get the following:
    
    ```python
    # Parameters
        learning_rate = 0.01
        training_epochs = 1000
        batch_size = 100
        display_step = 1
        with tf.Graph().as_default():
            # mnist data image of shape 28*28=784
            x = tf.placeholder("float", [None, 784])
    
            # 0-9 digits recognition => 10 classes
            y = tf.placeholder("float", [None, 10])
            output = inference(x)
            cost = loss(output, y)
            global_step = tf.Variable(0, name='global_step',
                                      trainable=False) # keeping track of epochs
            train_op = training(cost, global_step)
            eval_op = evaluate(output, y)
            summary_op = tf.merge_all_summaries()
            saver = tf.train.Saver()
            sess = tf.Session()
            summary_writer = tf.train.SummaryWriter("logistic_logs/",
                                      graph_def=sess.graph_def)
            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples/batch_size)
                # Loop over all batches
    
    						for i in range(total_batch):
    				                mbatch_x, mbatch_y = mnist.train.next_batch(
    				                    batch_size)
    				                # Fit training using batch data
    				                feed_dict = {x : mbatch_x, y : mbatch_y}
    				                sess.run(train_op, feed_dict=feed_dict)
    				                # Compute average loss
    				                minibatch_cost = sess.run(cost,
    				                    feed_dict=feed_dict)
    				                avg_cost += minibatch_cost/total_batch
    				            # Display logs per epoch step
    				     if epoch % display_step == 0:
    				                val_feed_dict = {
    				                     x : mnist.validation.images,
    				                     y : mnist.validation.labels
    				                }
    				                accuracy = sess.run(eval_op,
    				                    feed_dict=val_feed_dict)
    				                print "Validation Error:", (1 - accuracy)
    				                summary_str = sess.run(summary_op,
    				                                       feed_dict=feed_dict)
    				                summary_writer.add_summary(summary_str,
    				                                           sess.run(global_step))
    				                saver.save(sess, "logistic_logs/model-checkpoint",
    				                           global_step=global_step)
            print "Optimization Finished!"
            test_feed_dict = {
                 x : mnist.test.images,
                 y : mnist.test.labels
    }
            accuracy = sess.run(eval_op, feed_dict=test_feed_dict)
            print "Test Accuracy:", accuracy
    ```
    
    - Most of the code is self explanatory, create some constant variables and placeholders, loop through batch and epochs and train the model

### MNIST

```python
def layer(input, weight_shape, bias_shape):
        weight_stddev = (2.0/weight_shape[0])**0.5
        w_init = tf.random_normal_initializer(stddev=weight_stddev)
        bias_init = tf.constant_initializer(value=0)
        W = tf.get_variable("W", weight_shape,
                            initializer=w_init)
        b = tf.get_variable("b", bias_shape,
                            initializer=bias_init)
        return tf.nn.relu(tf.matmul(input, W) + b)
    def inference(x):
        with tf.variable_scope("hidden_1"):
            hidden_1 = layer(x, [784, 256], [256])
        with tf.variable_scope("hidden_2"):
						hidden_2 = layer(hidden_1, [256, 256], [256])
				with tf.variable_scope("output"):
						 output = layer(hidden_2, [256, 10], [10])
				return output
```

- There are many features of the error surfaces of deep neural networks that make optimization using vanilla stochastic gradient descent very difficult → smart initialization is one way to mitigate the issue
    - We changed `tf.random_normal_initializer` back to `tf.random_uniform_initializer` hurt the performance
        - We perform the softmax while computing the loss instead of during the inference phase of the network → leads into better performance

```python
def loss(output, y):
        xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y)
        loss = tf.reduce_mean(xentropy)
        return loss
```

- Running this gives improvement

### Chapter 4; Understanding Gradient Descent

- Spurious local minima corresponds to a configuration of weights in a neural net. that incurs a higher error than the configuration of the global min.
    - one way to naively tackle this problem is by plotting the value of the error function over time as we train a DNN
        - however this didn’t give enough information about the error surface because it’s difficult to tell whether the error surface is bumpy or whether we merely have a difficult time figuring out which direction we should be moving in
    - To more effectively do this, instead of analyzing the error function over time, they investigated what happens on the error surface between a randomly initialized parameter vector and a successful final solution by using linear interpolation
        - in other words, they wanted to see whether local minima would hinder our gradient-based search method even if we knew which direction to move in
- After using the code from before and constructing the linear interpolation, we get an output like this
    
    ![Screenshot 2023-07-24 at 7.48.10 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3650f7ba-b032-4eb3-ba5a-c496719acbfe/Screenshot_2023-07-24_at_7.48.10_PM.png)
    
    - In other words, there is no existence of a troublesome local minima, but instead we have a tough time finding the appropriate direction to move in

### Flat Regions in Error Surface

- at alpha = 1, we have a saddle point, it turns out as our function has more and more dimensions, saddle points are exponentially more likely than local minima
    - the reason for this is because in a d-dimensional space, we can slice through a critical point on d different axes, a critical point can only be a local min. if it appears as a local min in every single one of the d one-dimensional subspaces

### When Gradient Points in Wrong Direction

- when the contours of the gradient aren’t perfectly circular, the gradient can be as inaccurate as 90 degrees
    - the gradient gives us the direction of the steepest descent, but the issue is that the gradient could be changing under our feet as we move
- Hessian matrix
- when describing an error surface where the gradient changes under our feet, this matrix is said to be ill-conditioned
- certain properties of the hessian matrix allow us to determine the 2nd derivative as we move in a specific direction → if we have a unit vector d, we can now use a second order approximation via Taylor series to see what happens to the error function as we step from the current parameter vector to a new parameter vector along a gradient vector
    - computing hessian matrix is a bit difficult, but there are ways to avoid ill-conditioning without directly computing the Hessian matrix

### Momentum Based Optimization

- Ill conditioned hessian matrix manifests itself in the form of gradients that fluctuate wildly
    - If we look at how a ball rolls down a hilly surface, driven by gravity, it eventually settles into a minimum on the surface → it doesn’t suffer from the wild fluctuations and divergences that happen during gradient descent
        - the reason for this is, unlike stochastic GD, there are 2 major components that determine how the ball rolls down the error surface
            - acceleration + velocity
    - Velocity driven motion is desirable because it counteracts the effects of a wildly fluctuating gradient by smoothing the ball’s trajectory over its history
        - velocity serves as a form of memory, it allows us to more effectively accumulate movement in the direction of the min. while canceling out oscillating accelerations in orthogonal directions
            - We achieve this by keeping track of the history of past gradients (rate of change in the ball's position) in a weighted way. Every time we update the ball's position, we combine the update with the current gradient, allowing us to move more steadily towards the desired minimum point.
            - By doing this, the optimization algorithm behaves more like the ball rolling with velocity, making it less affected by the erratic fluctuations caused by the bumpy nature of the hill (or the ill-conditioned Hessian matrix, in mathematical terms). This helps the algorithm converge to the optimal solution more efficiently.
- we use the momentum hyper parameter to determine what fraction of the previous velocity to retain in the new update → momentum term increases the step size we take, using momentum may require a reduced learning rate
    
    ![Screenshot 2023-07-26 at 9.47.14 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9c399360-544e-441c-9322-7f2f57922dfb/Screenshot_2023-07-26_at_9.47.14_PM.png)
    

### Second Order Methods

- computing hessian matrix is very difficult → in order to get around this, some second order methods have been developed to help approx it
    - *Conjugate Gradient Descent* → it's an improvement over the simple steepest descent method. In steepest descent, we compute the direction of the gradient and move in that direction to find the minimum. However, this can lead to zigzagging and slow convergence. Conjugate gradient descent addresses this by choosing directions that are conjugate (related) to the previous ones, using an indirect approximation of the Hessian. This helps the optimization process work more efficiently, even for complex, non-convex error surfaces common in deep learning.
        
        ![Screenshot 2023-07-26 at 10.08.41 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/51faa599-22d0-46a3-964c-be8c66e8c8fb/Screenshot_2023-07-26_at_10.08.41_PM.png)
        
- Another option is *Broygen-Fletcher-Goldgarb-Shanno Algorithm*
    - Compute the inverse of the hessian matrix iteratively and uses the inverse to optimize the parameter vector

### Learning Rate Adaptation

- A learning rate too small doesn’t learn quickly enough → learning rate too large may struggle to converge to a local min. or region that is ill-conditioned
- AdaGrad, RMSProp, and Adam are the most popular adaptive learning rate algorithms

### AdaGrad

- this one attempts to adapt the global learning rate by using an accumulation of the historical gradients
    - keep track of a learning rate for each parameter → this learning rate is inversely scaled with respect to the square of the sum of the squares of all the parameters historical gradients
        - **Why Inversely Scaled**: By using the inverse square root of the sum of squared historical gradients, AdaGrad automatically reduces the learning rate for parameters that have received large gradients in the past. This helps prevent overshooting and taking excessively large steps in the parameter space, making the optimization process more stable.
- This all means that, the parameters with the largest gradients experience a rapid decrease in their learning rates, while parameters with smaller gradients only observe a small decrease
    - this forces more progress in the more gently sloped directions on the error surface, which can help overcome ill-conditioned surfaces
- however adagrad has a tendency to cause a premature drop in learning rate and doesn’t work well for deep models
    - flat regions may force adagrad to decrease the learning rate before it reaches a min.

### RMSProp

- Exponentially weighted moving averages allow us to toss out measurements that we made a long time ago → we can use RMSprop with momentum
    - this is very good for deep neural networks
- RMSprop calculates the exponential moving average of the squared gradients for each parameter. It means that instead of summing up all the squared gradients from the beginning of training, it takes a weighted average of the recent squared gradients. This way, the influence of older gradients diminishes over time.
- Analogy
    1. **Exponential Moving Averages**: To ensure proper watering, you keep a diary to track the amount of water you give each plant every day. However, instead of simply noting down the exact amount each time, you decide to take an average of the recent watering amounts for each plant. This average represents the moving average of the gradients in RMSprop.
    2. **Adaptive Watering**: Some plants may need more water than others, and their water requirements can change over time. RMSprop adjusts the amount of water (learning rate) for each plant (parameter) individually based on its historical watering amounts (gradients). If a particular plant has consistently received more water in the past, RMSprop reduces the amount of water given to prevent overwatering. Conversely, if a plant has received little water, RMSprop increases the amount to help it grow better.
    3. **Efficient Watering Management**: Since you are keeping track of recent watering amounts (moving averages) instead of recording the entire history, it becomes easier to manage the watering process efficiently. You don't need to store all the past watering data, just the most relevant recent information for each plant.
    4. **Bias Correction**: At the beginning of gardening, your diary is empty, and the moving averages might be biased. To avoid this issue, you perform bias correction by adjusting the moving averages based on the number of days you have been gardening. This ensures that the moving averages get a fair start before you make more adjustments.

### Adam

- this is viewed as a combo of RMSProp and momentum
    - we want to keep track of an exponentially weighted moving average of the gradient
1. **Exponential Moving Averages**: Adam maintains two exponential moving averages of the gradients. One moving average keeps track of the first moment (mean) of the gradients, and the other keeps track of the second moment (uncentered variance) of the gradients.
2. **Bias Correction**: Since the moving averages are initialized at zero, they can be biased, especially at the beginning of training. To counteract this bias, Adam uses bias correction by scaling the moving averages based on the number of iterations performed.
3. **Combining Momentum and Moving Averages**: Adam combines the momentum term and the moving averages to update the model's parameters. The adaptive learning rates, as well as the momentum, influence the step taken in the parameter space.
- Analogy
    1. **Exponential Moving Averages**: To make the perfect soup, you want to keep track of how the flavor changes over time as you add different spices. You maintain two bowls: one to calculate the average intensity of flavors (mean) and another to calculate the average "kick" or "punch" of flavors (variance). Each time you add a spice, you take a small sample of the flavor intensity and punch it adds to the soup and add it to the corresponding bowl.
    2. **Bias Correction**: At the beginning of cooking, both bowls are empty, and the flavor intensity and punch averages might be biased due to a lack of data. To overcome this, you perform bias correction. It's like giving the bowls a few initial stirs, so they have some reasonable estimates to start with before you start adding more spices (iterations).
    3. **Combining Momentum and Moving Averages**: As you taste the soup and adjust the flavor, you don't want to make drastic changes with each new spice. Instead, you want to consider how the flavor has been changing in the past and make gradual adjustments. You use a spoon (momentum) to mix the contents of the two bowls (the moving averages). The momentum helps you remember the direction and strength of previous changes in flavor. When you add a new spice, you take a spoonful from each bowl and combine them to determine how much to adjust the soup's taste.

## Chapter 5: Convolutional Neural Networks

### Shortcomings of Feature Selection

- ML algorithms use feature vectors to make classification decisions → the feature extraction process improves the signal-to-noise ratio
    - Viola and Jones (they tackled the problem of identifying whether a face is human or not in 2001) had the insight that faces had certain patterns of light and dark patches that they could exploit
        - combining features together helps improve the accuracy of the model
- However there were some down sides, such as, if a face was covered with some shade, the light intensity comparisons no longer work
    - the algo. hadn’t learnt much beyond differences in light intensity, our brain uses a vast number of visual cues to realize that our field of view contains a human face
- In order for ML techniques to teach a computer to see, we need to provide the program with more features to make accurate decisions
- AlexNet was created in Geoffrey Hinton’s lab at UofT with the proposed idea of a CNN and beat all the competition in the ImageNet challenge

### Vanilla Deep NNs don’t scale

- the main idea of applying deep learning to CV is to remove the feature selection process
    - a naive approach would be to use a vanilla deep neural network using the network layer primitive for the MINST dataset
- in MNIST, the images were only 28 x 28 = 784 incoming weights → but this doesn’t scale well for larger input images → this would mean that we’re likely to overfit to the training dataset due to large number of parameters
- CNNs take advantage of the fact that we’re looking at images and constrains the model of the deep network so that we reduce the number of parameters in our model
    - layers of a CNN have neurons arranged in 3 dimensions so layers have width, height, and depth
    - neurons in a convolutional layer are only connected to a small, local region of the preceding layer → helps avoid the wastefulness of fully connected neurons

### Filters and Feature Maps

- in the human brain, parts of the visual cortex is responsible for detecting edges → they found that some neurons fired only when there were vertical lines, others fired when there were horizontal lines
    - they eventually discovered that the visual cortex was organized in layers → each layer is responsible for building on top of the features detected in the previous layers
        - the first concept that arose was that of a filter → a filter is essentially a feature detector
            
            ![Screenshot 2023-08-02 at 3.44.25 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d767577a-df3f-47f9-9043-3b1ba532f3a9/Screenshot_2023-08-02_at_3.44.25_PM.png)
            
- Consider the image above, let’s say we want to detect vertical and horizontal lines → one approach would be to use a feature detector
    - feature detector on the top, slide it across the entirety of the image and at every step, check if we have a match
        - we keep track of our answers in the matrix in the top right, if there’s a match we shade the appropriate black box, else, leave it white. the result is called a feature map, and it indicates where we’ve found the feature we’re looking for in the original image
        
        ![Screenshot 2023-08-02 at 3.45.23 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d7c91884-9c32-4f31-a4a4-9db41fc71550/Screenshot_2023-08-02_at_3.45.23_PM.png)
        
- this operation is called a *convolution*
    - we take a filter and we multiply it over the entire area of an input image
- Layers of neurons in a feed-forward neural net represent the original image or a feature map
    - filters represent combinations of connections that get replicated across the entirety of the input
    
    ![Screenshot 2023-08-02 at 3.49.36 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f688e6c7-d0b3-4d99-b4d5-d994929dbad7/Screenshot_2023-08-02_at_3.49.36_PM.png)
    
- connections of the same colour have the same weight → To ensure this, we start by setting all connections of the same color with the same weight. Then, when we make adjustments to the weights during training (using a process called backpropagation), we calculate the average change for all the connections in the same group. This way, they stay equal, and the filter doesn't become biased towards one color over others.
    - output layer is the feature map
        - neuron in the feature map is activated if the filter contributing to its activity detected an appropriate feature at the corresponding position in the previous layer
- mk = f(W * x + bk)
    - In this equation, "x" represents the input data, and "W * x" is the dot product of the filter weights and the input data. The function "f" represents an activation function that determines whether the neuron should be activated or not based on its input.
- However, the above mathematical description is simplified and doesn't fully describe how filters work in convolutional neural networks. Filters don't operate on just one feature map; they work on the entire volume of feature maps that are generated at a particular layer.
- For example, imagine you want to detect a face using a convolutional neural network. At a specific layer of the network, you might have three feature maps: one for eyes, one for noses, and one for mouths. To determine if there is a face, you need to combine evidence from all three feature maps. If all the appropriate features (two eyes, a nose, and a mouth) are present at the corresponding locations in these feature maps, then a face is detected.
- This concept is essential for processing full-color images represented as RGB values. Each color channel requires its own slice in the input volume, and feature maps must be able to operate over volumes rather than just individual areas.
    
    ![Screenshot 2023-08-02 at 3.56.40 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7c51ee57-5b43-4033-8e47-a6f82981794a/Screenshot_2023-08-02_at_3.56.40_PM.png)
    

### Convolutional Layer

- a convolutional layer takes in an input volume → it has the following characteristics
    - weight, height, depth, zero padding
- Volume is processed by a total of k filters, which represent the weights and connections in the convolutional network
    - the filters have the following hyper-parameters
        - spatial extent e = filter’s height and width
        - stride s = distance between the consecutive applications of the filters on the input volume. if we use stride of 1, we get the full convolution
            
            ![Screenshot 2023-08-02 at 4.12.58 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/68d356bd-38a2-423b-9b1f-af3955d18ba1/Screenshot_2023-08-02_at_4.12.58_PM.png)
            
        - bias b = which is added to each component of the convolution
- all of that results in an output volume with the following characteristics
    - its function f, which is applied to the incoming logit of each neuron in the output volume to determine its final value
    - width
    - height
    - depth
        
        ![Screenshot 2023-08-02 at 4.14.40 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/319a293d-cad4-4233-b48a-9ff8df1b0c1d/Screenshot_2023-08-02_at_4.14.40_PM.png)
        
- it’s recommended to keep filter sizes small, having small filters is an easy way to achieve high representational power while also incurring a smaller number of parameters
    - use stride of 1 to capture all useful information and a zero padding that keeps the output volume’s height and width equivalent to the input volume’s height + width

### Max Pooling

- to reduce the dimensions of a feature map and sharpen the located features, we sometimes insert a max pooling layer after a convolutional layer
    - the idea is to break up each feature map into equally sized tiles and create a condensed feature map
        - we compute the max value in the tile and propagate this max value into the corresponding cell of the condensed feature map
            
            ![Screenshot 2023-08-03 at 11.54.12 AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ef2268dc-00ca-431c-abc5-259452fbbc1d/Screenshot_2023-08-03_at_11.54.12_AM.png)
            
- We can describe the pooling layer w/ 2 parameters
    - spatial extent e
    - stride s
- the results are the following
    - width
    - height
- Max pooling is locally invariant → which means that even if the inputs shift around a bit, the output stays constant

### Understanding full architecture

![Screenshot 2023-08-03 at 11.57.07 AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7a82d677-ddaf-4589-b7ba-397236b7d3d4/Screenshot_2023-08-03_at_11.57.07_AM.png)

- one theme we see as we build deeper networks is that we reduce the number of pooling layers and instead stack multiple convolutional layers in tandem
    - this is helpful b/c pooling operations are destructive, stacking convolutional layers helps us get a more meaningful representation

### Shifting back to MNIST

- let’s build a CNN
    
    ```python
    def conv2d(input, weight_shape, bias_shape):
             in = weight_shape[0] * weight_shape[1] * weight_shape[2]
             weight_init = tf.random_normal_initializer(stddev=
                                                       (2.0/in)**0.5)
             W = tf.get_variable("W", weight_shape,
                                 initializer=weight_init)
             bias_init = tf.constant_initializer(value=0)
             b = tf.get_variable("b", bias_shape, initializer=bias_init)
             conv_out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1],
                                   padding='SAME')
             return tf.nn.relu(tf.nn.bias_add(conv_out, b))
      def max_pool(input, k=2):
             return tf.nn.max_pool(input, ksize=[1, k, k, 1],
                                   strides=[1, k, k, 1], padding='SAME')
    ```
    
- First method generates a conv. layer with a particular shape
- the second method generates a max pooling layer with non-overlapping windows of size k → default is k =2 so we use that
- Here is the final code
    
    ```python
    def inference(x, keep_prob):
            x = tf.reshape(x, shape=[-1, 28, 28, 1])
            with tf.variable_scope("conv_1"):
                conv_1 = conv2d(x, [5, 5, 1, 32], [32])
                pool_1 = max_pool(conv_1)
            with tf.variable_scope("conv_2"):
                conv_2 = conv2d(pool_1, [5, 5, 32, 64], [64])
                pool_2 = max_pool(conv_2)
            with tf.variable_scope("fc"):
                pool_2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64])
                fc_1 = layer(pool_2_flat, [7*7*64, 1024], [1024])
                # apply dropout
                fc_1_drop = tf.nn.dropout(fc_1, keep_prob)
            with tf.variable_scope("output"):
                output = layer(fc_1_drop, [1024, 10], [10])
            return output
    ```
    
    - First we take the flattened versions of the input pixel values and reshape them
        - depth = 1 as images are black and white, it would be 3 if it was rgb
        - we then build a convolutional layer with 32 filters that have spatial extent 5
            - this results in taking an input volume of depth 1 and emitting a output tensor of depth 32
            - this is then passed through a max pooling layer which compresses the information
        - we take a 2nd conv. layer with 64 layers, spatial extent 5, taking an input tensor of depth 32 and emitting an output tensor of depth 64
            - this is again passed through max pooling
        - we then prepare to pass the output of the max pooling into a fully connected layer → to do this, we flatten the tensor (compute the full size of each subtensor in the minibatch)
            - we now have to determine the height and width after passing through 2 max pooling layers → it’s easy to confirm that each feature map has a height and width of 7
        - after reshaping, we use a fully connected layer to compress the flattened representation into a hidden state of size 1024 → we use a dropout probability in this layer of 0.5 during training and 1 during model evaluation
            - this is the standard
        - finally we train with the adam optimizer and we achieve accuracy of 99.4%

### Batch Normalization

- This is a way to further accelerate the training of feed-forward and CNNs
- Analogy with tower of blocks
    - When a tower of block sis stacked together neatly, the structure is stable, but if the blocks are shifted, they are increasingly more nunstable
    - This happens during the training of neural networks, imagine a 2-layer neural net
        - in the process of training the weights of the network, the output distribution of neurons in the bottom layer begins to shift
            - the result of the changing distribution of outputs from the bottom layer means that the top layer  not only has to learn how to make the predictions, but also needs to somehow modify itself to accommodate the shifts in incoming distribution → this slows down training + the magnitude of the problem compounds the more layers we have in our network
- Normalization of image inputs helps the training process by making it more robust to variations
    - batch normalization normalizes inputs to every layer in our network → specifically:
        1. grab the vector of logits incoming to a layer before they pass through the nonlinearity
        2. normalize each component of the vector of logits across all examples of the minibatch
        3. given normalizes input x, use an affine transform to restore representational power w/ 2 vectors of trainable parameters
- batch normalization allows us to increase the learning rate + acts as a normalizer and removes the need for dropout

### Visualizing Learning in CNNs

- simplest thing to do is plot the cost function + validation errors over time as training progresses

## Chapter 7

### RNNs

- They are feed-forward networks because they leverage a special type of neural layer → *recurrent layers*
- They enable the network to maintain state between uses of the network
- All of the neurons have the following:
    - incoming connections emanating from all of the neyrons in the previous layer
    - outgoing connectoins leading to all of the neurons to the subsequent layer
- recurrent layers also have recurrent connections → these propagate information between neurons of the same layer
- A fully connected recurrent layer has information flow from every neuron to every other neurons in its layer (including itself)
    
    ![Screenshot 2023-08-07 at 7.14.22 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/848d5521-11c8-4c3a-b4a6-6414e4f9f1d4/Screenshot_2023-08-07_at_7.14.22_PM.png)
    
- every time we want to process a new sequence, we create a new instance of the model
    - at each time step, we feed the model the next element of the input
    - feedforward connections represent information flow from one neuron to another where the data being transferred is the computed neuronal activation from the current time step
    - recurrent connections represent information flow where the data is the stored activation from the previous time step → the activations of the neurons in a recurrent network represent the accumulating state of the network instance
- The initial activations are like parameters, we determine optimal values for them just like weights

---

- if we give the model a fixed lifetime (for ex. *t* steps), we can express the instance as a feed-forward network
    - this is referred to as unrolling the RNN through time
- Easier Understanding of Unrolling
    - To make it easier to understand, we can pretend that the RNN runs for a fixed number of steps, let's say t steps. Then, we create t copies of the RNN, one for each time step, like making t identical copies of the program.
    - Next, we connect these copies together so that the output of one copy becomes the input of the next copy. It's like passing information from one step to the next, just like the original RNN does with its recurrent connections.
    - Now, we have a series of connected copies of the RNN, and it looks more like a regular feed-forward network, which is easier to understand and work with.
    - This transformation is called "unrolling" the RNN through time, and it helps us see how the RNN processes sequences step by step. It's like looking at the RNN's behavior over time as if it were a regular network with multiple layers.

![Screenshot 2023-08-07 at 7.32.28 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6c03ddcf-6f8d-4a57-9a37-a904f561e3be/Screenshot_2023-08-07_at_7.32.28_PM.png)

- we can now also train the RNN by computing the gradient based on the unrolled version
    - this means that all of the backprop. techniques that we utilized for feedforward can be used here
        - after each batch of training examples, we need to modify the weights based on the error derivatives we calculate
            - in our unrolled network, we have sets of connections that all correspond to the same connection in the original RNN
                - the error derivatives aren’t guaranteed to be equal
- to get around this, we can average or sum the error derivatives over all the connections

### Vanishing gradients!?!

1. The RNN has a memory bank, represented by a recurrent layer, which allows it to remember information from previous steps and summarize the dependencies between elements in the input sequence.
2. In theory, researchers Kilian and Siegelmann showed in 1996 that an RNN is a universal functional representation. It means that, with enough neurons (processing units) and appropriate settings, an RNN can be used to represent any kind of relationship between input and output sequences.
3. However, while the theory is promising, it might not always work well in practice. Even though an RNN has the potential to represent any function, we need to know if it's practical to teach the RNN to learn realistic relationships between input and output sequences from scratch.
4. To find out if it's practical, we need to use a technique called gradient descent algorithms. These algorithms help us adjust the parameters of the RNN to make it learn the desired relationship between sequences.
- This issue is known as vanishing gradients
    - The problem arises when the gradients (slopes) of the network's error with respect to the model's parameters become extremely small as they are propagated backward through time during the training process.
    - During the training of an RNN, the network tries to adjust its internal parameters to minimize the error between the predicted output and the actual target output. This adjustment is done using a technique called backpropagation, which involves calculating gradients and updating the parameters based on these gradients.
    - The issue with vanishing gradients occurs when the gradients become very close to zero as they are propagated from the output back to the earlier time steps in the sequence. This happens because of the repeated multiplication of small values (often associated with activation functions like sigmoid or tanh) during the backpropagation process.
    - When the gradients become too small, it becomes challenging for the RNN to learn long-term dependencies in the data. Essentially, the network is not able to effectively adjust its parameters to capture information from earlier time steps, leading to poor performance and difficulty in learning patterns that span over many time steps.
- To get around vanishing gradients, an LSTM model was proposed

### LSTM (long short-term memory)

- the basic idea behind this architecture is that the network would be designed for the purpose of reliably transmitting important info many time steps into the future
    
    ![Screenshot 2023-08-08 at 2.43.52 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f959a93a-8fc3-4333-82bc-d8857cde4399/Screenshot_2023-08-08_at_2.43.52_PM.png)
    
- LSTM unit is composed of a memory cell → a tensor represented by the bolded loop int the center
    - the memory cell holds critical information that is learned over time and the network is designed to effectively maintain useful information over many time steps
        - at every time step, the LSTM model modifies the memory cell with new information with 3 diff phases
    - First the unit must determine how much of the previous memory to keep → this is determined by the keep gate
- Keep gate
    
    ![Screenshot 2023-08-08 at 5.37.55 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8bde35aa-7aa5-4cca-90fd-68fe99c5c070/Screenshot_2023-08-08_at_5.37.55_PM.png)
    
    - the memory state tensor from the previous time step has lots of information, but some of that info is stale/useless
        - we figure out which elements are still relevant and which aren’t by computing a bit tensor (tensor of 0s and 1s) that we multiply w the previous state
    - if a particular location in the bit tensor holds a 1, it means it is still relevant, but if it’s a 0, it means it’s irrelevant
    - to approximate the bit tensor → concatenating the input of this time step and the LSTM output from the previous time step and applying a sigmoid layer to the resulting tensor
- Memory state / write gate
    
    ![Screenshot 2023-08-08 at 5.38.04 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f4e294b3-a6cd-4c71-ae26-1aa2f3d05543/Screenshot_2023-08-08_at_5.38.04_PM.png)
    
    - this is broken down into 2 components → first one is responsible for figuring out what information to write into the state
        - this is done by computing the `tanh` layer, to create an intermediate tensor
    - the second component is figuring out which component of this computed tensor do we actually want to include into the new state and which we want to toss before writing
        - we do this by approx. a bit vector of 0’s and 1’s using the same strategy used in the keep gate
- Output gate
    
    ![Screenshot 2023-08-08 at 5.41.04 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/565fc93f-1ab4-4a2f-9ca5-230a2b5e0727/Screenshot_2023-08-08_at_5.41.04_PM.png)
    
    - Finally, at each time step, we want the LSTm to provide an output → we could treat the state vector as the output directly, but the LSTM is engineered to provide more flexibility by emitting an output tensor that is a “interpretation” or extrenal “communication” of what the state vector represents
    - the architecture is as follows
        - the tanh layer creates an intermediate tensor
        - sigmoid layer produces a bit tensor mask
        - intermediate tensor is multiplied w/ bit tensor to produce final output