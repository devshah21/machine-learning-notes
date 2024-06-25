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
