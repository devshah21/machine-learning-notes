# optimizers + regularization

## optimizers.

- optimizers are used to adjust the parameters of a model — this is done to help minimize the loss function (this is what helps the model actually learn)
- **purpose of optimizers**
    - minimize loss
    - efficiency: optimizers help in navigating the parameter space to find the minimum loss in a reasonable amount of time
    - convergence: they ensure that the training process converges to a solution — best case scenario, a global minimum, but average case, a good local minimum
- **mini-batch gradient descent**
    - this approach updates the weights after seeing **N** samples — original gradient descent would just update the parameters after seeing the entire dataset, this approach is better since we’re updating after every couple samples, which allows for better + quicker convergence
    - this is how it works:
        1. shuffle the data
        2. compute the gradient
            1. compute the prediction
            2. calculate the loss
            3. compute the gradient of the loss with respect to the model parameters
        3. parameter update
            1. $\theta = \theta - \eta \cdot \nabla_{\theta} L(\hat{y}_i, y_i)$
            
    - pseudo code for this approach
        
        ```python
        for epoch in range(num_epochs):
            shuffle(training_data)
            for x_i, y_i in training_data:
                prediction = model(x_i)
                loss = loss_function(prediction, y_i)
                gradient = compute_gradient(loss, model.parameters)
                model.parameters -= learning_rate * gradient
        ```
        
- **sgd with momentum**
    - the idea here is that the momentum parameters helps accelerate convergence, especially when there’s a lot of oscillations in the loss function
        - the way that this is done is through a velocity vector
    - intuition:
        - acceleration in the right direction: by accumulating past gradients, momentum helps to accelerate the updates in directions where the gradient consistently points, thus this **speeds up** convergence
        - smoothing oscillations: in directions where the gradient changes sign frequently, momentum helps smooth out these oscillations, leading to more stable convergence
    - how it works:
        1. velocity update
            1. the velocity vector is updated as a combination of the previous velocity & the current gradient
                1. the parameter, beta, is the momentum coefficient and it controls how much of the previous velocity is retrained
                    1. $v_t = \beta v_{t-1} + (1 - \beta) \nabla_{\theta} L(\theta)$
        2. parameter update
            1. the parameters are then updated using this velocity vector instead of the raw gradient
                1. $\theta = \theta - \eta \cdot v_t$
    - psuedocode
        
        ```python
        # Initialize parameters
        theta = initial_parameters()
        v = 0  # Initial velocity
        beta = 0.9  # Momentum coefficient
        eta = 0.01  # Learning rate
        
        for epoch in range(num_epochs):
            shuffle(training_data)
            for x_i, y_i in training_data:
                prediction = model(x_i)
                loss = loss_function(prediction, y_i)
                gradient = compute_gradient(loss, model.parameters)
                
                # Update velocity
                v = beta * v + (1 - beta) * gradient
                
                # Update parameters
                model.parameters -= eta * v
        ```
        

## regularization.

- in short, this is basically a bunch of methods to **reduce overfitting** in machine learning models
    - there’s a trade off that is made here → regularization trades a slight decrease in training accuracy for better generalization for the model
- **understand bias vs. variance**
    - bias measures the avg difference between the predicted outcome vs. the true value
        - when bias increases — model training accuracy goes down (high bias ⇒ poor training accuracy)
    - variance measures the difference between predictions across various realizations of a given model
        - as variance increases, the model predicts less accurately on unseen data (high variance ⇒ high error during testing + val.)
- **general idea of regularization**
    - by increasing the bias and decreasing variance, regularization solves the problem of model overfitting
    - **key idea:** overfitting is basically when the model has low bias and high variance
        - under-fitting: high bias & high variance
- **types of regularization**
    - L1 regularization (Lasso regression)
        - this technique penalizes high-value, correlated coefficients
            - it introduces a regularization term (penalty term) into the model’s MSE loss function — this term is the absolute value of the sum of coefficients
                - it is controlled by a hyperparameter $\lambda$, the equation is: $L = L_{original} + \lambda \sum_{j=1}^p |w_j|$
                    - the term inside the summation represents the coefficients of the features and p is the number of features
            - this sort of acts as a penalty since it constrains the coefficients from becoming too large
        - L1 regularization produces sparse models — which means some coefficients are driven exactly to 0 — which effectively removes certain features from the model (in order to minimize loss)
        
        ```python
        from sklearn.linear_model import Lasso
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import fetch_california_housing
        from sklearn.metrics import mean_squared_error
        
        data = fetch_california_housing()
        X = data.data
        y = data.target
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # define the L1 / lasso regression model with a chosen alpha (λ)
        lasso = Lasso(alpha=0.1)
        
        lasso.fit(X_train, y_train)
        
        y_pred = lasso.predict(X_test)
        
        # eval the model
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')
        
        print('Coefficients:', lasso.coef_)
        ```
        
    - L2 regularization (Ridge regression)
        - it does something similar to the L1 loss, but it penalizes high-value coefficients in SSE  loss function
            - it’s a bit different as the penalty term is the squared sum of coefficients rather than the absolute value
                - L2 regularization also never sets features to 0, only towards 0
            - here is the equation representation: $L = L_{og} + \lambda \sum_{j=1}^pw_j^2$
                - lambda is the regularization parameter that controls strength of the penality
                - $w_j$ represents coefficients of the features
                - p is the number of features
            - the main difference between L1 and L2 is that L2 shrinks the features towards 0 which reduces their influence, but they are still retained
            - L2 regularization produces more stable & less sensitive models to small changes in the training data
            
            ```python
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import train_test_split
            from sklearn.datasets import fetch_california_housing
            from sklearn.metrics import mean_squared_error
            
            # Load dataset
            data = fetch_california_housing()
            X = data.data
            y = data.target
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Define Ridge regression model with a chosen alpha (λ)
            ridge = Ridge(alpha=1.0)
            
            # Fit the model to the training data
            ridge.fit(X_train, y_train)
            
            # Predict on the test data
            y_pred = ridge.predict(X_test)
            
            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            print(f'Mean Squared Error: {mse}')
            
            # Print the coefficients
            print('Coefficients:', ridge.coef_)
            
            ```
            
    - Elastic Net Regularization
        - this basically combines both L1 + L2 regularization penalty terms into the SSE loss function
            - here is the equation: $L = L_{og} + \alpha \left( \lambda_1 \sum_{j=1}^{p} |w_j| + \lambda_2 \sum_{j=1}^{p} w_j^2 \right)$
                - alpha is the overall regularization strength with respective lambdas
            - since L1 produces sparse models and L2 keeps all coefficients above 0, this regularization essentially benefits from both sparsity & ability to handle correlated features
            - for example, when we have a dataset with highly correlated features:
                - L1 regularization (Lasso) can arbitrarily select one feature from a group of correlated features and discard the others, which may not be desirable. L2 regularization (Ridge) shrinks the coefficients of correlated features together but doesn't perform feature selection. Elastic Net, by combining both L1 and L2 penalties, can include groups of correlated features and perform feature selection more effectively.
            
            ```python
            from sklearn.linear_model import ElasticNet
            from sklearn.model_selection import train_test_split
            from sklearn.datasets import fetch_california_housing
            from sklearn.metrics import mean_squared_error
            
            # Load dataset
            data = fetch_california_housing()
            X = data.data
            y = data.target
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Define Elastic Net model with chosen alpha and l1_ratio (λ1/λ2)
            elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio=0.5 means equal balance between L1 and L2
            
            # Fit the model to the training data
            elastic_net.fit(X_train, y_train)
            
            # Predict on the test data
            y_pred = elastic_net.predict(X_test)
            
            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            print(f'Mean Squared Error: {mse}')
            
            # Print the coefficients
            print('Coefficients:', elastic_net.coef_)
            
            ```