import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Simulated dataset
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Number of folds
k = 5

# Initialize KFold
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize an array to store accuracy scores
accuracy_scores = []

# Perform k-fold cross validation
for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
    # Split the data
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    accuracy_scores.append(accuracy)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")

# Calculate and print the average accuracy
average_accuracy = np.mean(accuracy_scores)
print(f"\nAverage Accuracy: {average_accuracy:.4f}")