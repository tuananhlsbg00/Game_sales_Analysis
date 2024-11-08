import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('MAS291 Project.csv')

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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train, test, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the model using Ridge Regression
ridge_model = Ridge(alpha=1.0)  # You can adjust the regularization strength with the alpha parameter
ridge_model.fit(X_train, y_train)

# Make predictions on train set
prediction_train = ridge_model.predict(X_train)

# Make predictions on test set
predictions_test = ridge_model.predict(X_test)

# Make predictions on validation set
predictions_val = ridge_model.predict(X_val)

# Calculate Mean Squared Error on training set
mse_train = np.mean((y_train - prediction_train) ** 2)
print("Mean Squared Error on training Set:", mse_train)

# Calculate Mean Squared Error on test set
mse_test = np.mean((y_test - predictions_test) ** 2)
print("Mean Squared Error on Test Set:", mse_test)

# Calculate Mean Squared Error on validation set
mse_val = np.mean((y_val - predictions_val) ** 2)
print("Mean Squared Error on Validation Set:", mse_val)

# Calculate R-squared on train set
r_squared_train = 1 - (np.sum((y_train - prediction_train) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2))
print("R-squared on Train Set:", r_squared_train)

# Calculate R-squared on test set
r_squared_test = 1 - (np.sum((y_test - predictions_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
print("R-squared on Test Set:", r_squared_test)

# Calculate R-squared on validation set
r_squared_val = 1 - (np.sum((y_val - predictions_val) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))
print("R-squared on Validation Set:", r_squared_val)

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
