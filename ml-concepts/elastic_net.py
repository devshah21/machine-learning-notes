from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio=0.5 means equal balance between L1 and L2


elastic_net.fit(X_train, y_train)

y_pred = elastic_net.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

print('Coefficients:', elastic_net.coef_)
