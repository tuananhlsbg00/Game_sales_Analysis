import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('ExculdeNA.csv')

# Fill missing values in numerical columns with the mean
data['Critic_Score'].fillna(data['Critic_Score'].mean(), inplace=True)
data['Critic_Count'].fillna(data['Critic_Count'].mean(), inplace=True)

# Fill missing values in categorical columns with the most common value
data['Name'].fillna(data['Name'].mode()[0], inplace=True)
data['Platform'].fillna(data['Platform'].mode()[0], inplace=True)
data['Year_of_Release'].fillna(data['Year_of_Release'].mode()[0], inplace=True)
data['Genre'].fillna(data['Genre'].mode()[0], inplace=True)
data['Rating'].fillna(data['Rating'].mode()[0], inplace=True)

# Select features and target variable
X = data[['NA_Sales']]
y = data['Global_Sales']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into train, test, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the train set
loss_train = model.evaluate(X_train, y_train)
print("Mean Squared Error on Train Set:", loss_train)

# Evaluate the model on the test set
loss_test = model.evaluate(X_test, y_test)
print("Mean Squared Error on Test Set:", loss_test)

# Evaluate the model on the validation set
loss_val = model.evaluate(X_val, y_val)
print("Mean Squared Error on Validation Set:", loss_val)

# Make predictions on training set
predictions_train = model.predict(X_train)

# Calculate R-squared on training set
r_squared_train = r2_score(y_train, predictions_train)
print("R-squared on Train Set:", r_squared_train)

# Make predictions on test set
predictions_test = model.predict(X_test)

# Calculate R-squared on test set
r_squared_test = r2_score(y_test, predictions_test)
print("R-squared on Test Set:", r_squared_test)

# Make predictions on validation set
predictions_val = model.predict(X_val)

# Calculate R-squared on validation set
r_squared_val = r2_score(y_val, predictions_val)
print("R-squared on Validation Set:", r_squared_val)

# Plot loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Function to plot scatter plot of dataset and prediction line
def plot_scatter_with_prediction(X_test, y_test, model):
    # Plot scatter plot of test data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_test[:, 0], y_test, label='Actual', color='blue', alpha=0.5)
    
    # Predict global sales using the model
    predictions = model.predict(X_test)
    
    # Plot prediction line
    ax.scatter(X_test[:, 0], predictions, label='Predicted', color='red', alpha=0.5)
    
    # Set labels for each feature
    feature_labels = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score', 'Critic_Count']
    
    # Set labels for x and y axes
    ax.set_xlabel('Features')
    ax.set_ylabel('Global_Sales')
    ax.set_title('Scatter Plot with Prediction Line')
    
    # Add legend
    ax.legend()
    
    plt.show()

# Plot scatter plot with prediction line for test set
plot_scatter_with_prediction(X_test, y_test, model)
