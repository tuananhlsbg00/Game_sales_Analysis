import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv('ExculdeNA.csv')

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            # Calculate and store the loss
            loss = self.mse(y, y_predicted)
            self.losses.append(loss)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def r_squared(self, y_true, y_pred):
        # Calculate the mean of the true target values
        y_mean = np.mean(y_true)
        # Calculate the total sum of squares
        tss = np.sum((y_true - y_mean) ** 2)
        # Calculate the residual sum of squares
        rss = np.sum((y_true - y_pred) ** 2)
        # Calculate R-squared
        r_squared = 1 - (rss / tss)
        return r_squared
    
    def plot_loss_convergence(self, losses):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-')
        plt.title('Loss Function Convergence')
        plt.xlabel('Iterations')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)
        plt.show()

# Fill missing values in numerical columns with the mean
data['Critic_Score'].fillna(data['Critic_Score'].mean(), inplace=True)
data['Critic_Count'].fillna(data['Critic_Count'].mean(), inplace=True)

# Fill missing values in categorical columns with the most common value
data['Name'].fillna(data['Name'].mode()[0], inplace=True)
data['Platform'].fillna(data['Platform'].mode()[0], inplace=True)
data['Year_of_Release'].fillna(data['Year_of_Release'].mode()[0], inplace=True)
data['Genre'].fillna(data['Genre'].mode()[0], inplace=True)
data['Rating'].fillna(data['Rating'].mode()[0], inplace=True)

X = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Critic_Score']].values
y = data['Global_Sales'].values

# Normalize the features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

# Split the data into train, test, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X_normalized, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the model using the training set
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on train set
prediction_train = model.predict(X_train)

# Make predictions on test set
predictions_test = model.predict(X_test)

# Make predictions on validation set
predictions_val = model.predict(X_val)

# Calculate Mean Squared Error on training set
mse_train = model.mse(y_train, prediction_train)
print("Mean Squared Error on training Set:", mse_train)

# Calculate Mean Squared Error on test set
mse_test = model.mse(y_test, predictions_test)
print("Mean Squared Error on Test Set:", mse_test)

# Calculate Mean Squared Error on validation set
mse_val = model.mse(y_val, predictions_val)
print("Mean Squared Error on Validation Set:", mse_val)

# Calculate R-squared on train set
r_squared_train = model.r_squared(y_train, prediction_train)
print("R-squared on Train Set:", r_squared_train)

# Calculate R-squared on test set
r_squared_test = model.r_squared(y_test, predictions_test)
print("R-squared on Test Set:", r_squared_test)

# Calculate R-squared on validation set
r_squared_val = model.r_squared(y_val, predictions_val)
print("R-squared on Validation Set:", r_squared_val)

# Plot loss function convergence
model.plot_loss_convergence(model.losses)

# Plot scatter plots of all features against the target variable
plt.figure(figsize=(12, 8))

for i, feature in enumerate(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Critic_Score']):
    plt.subplot(2, 2, i+1)
    plt.scatter(X_test[:, i], y_test, color='blue', label='Actual Test')
    plt.scatter(X_test[:, i], predictions_test, color='red', label='Predicted Test')
    plt.scatter(X_val[:, i], y_val, color='green', label='Actual Validation')
    plt.scatter(X_val[:, i], predictions_val, color='yellow', label='Predicted Validation')
    plt.xlabel(feature)
    plt.ylabel('Global_Sales')
    plt.title(f'Scatter Plot of Global Sales vs {feature}')
    plt.legend()

plt.tight_layout()
plt.show()
