import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

start = time.time()
# --- Load and Preprocess Dataset ---

# Load the dataset
data = pd.read_csv('Train_Data.csv')

# Fill missing values
num_imputer = SimpleImputer(strategy='mean')  
cat_imputer = SimpleImputer(strategy='most_frequent')

num_cols = ['NA_Sales','Critic_Score', 'Critic_Count']#, 'EU_Sales', 'JP_Sales']#, 'Critic_Score', 'Critic_Count']
cat_cols = ['Genre']
# num_cols = ['Critic_Score']

# Create a pipeline to handle both numerical and categorical data
preprocessor = ColumnTransformer([
    ('num_cols', Pipeline([('imputer', num_imputer), ('scaler', StandardScaler())]), num_cols),
    ('cat_cols', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# preprocessor = ColumnTransformer([
#     ('num_cols', Pipeline([('imputer', num_imputer), ('scaler', StandardScaler())]), num_cols)
# ])

X = data.drop('Exclude NA', axis=1)
y = data['Exclude NA']

X = preprocessor.fit_transform(X).toarray()  # Convert to dense NumPy array

# --- Split into Train, Test, and Validation Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=150)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=56)

# --- Create the Neural Network ---

model = Sequential([
    Dense(32, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(1) 
])


model.compile(optimizer='Adam', loss='mse', metrics=['mae'])  # Add 'mae' if desired

# --- Train the Model ---
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), batch_size=80)



# --- Evaluate Performance ---
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, r2

train_mse, train_r2 = evaluate_model(model, X_train, y_train)
# val_mse, val_r2 = evaluate_model(model, X_val, y_val)
test_mse, test_r2 = evaluate_model(model, X_test, y_test)

print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
# print("Validation MSE:", val_mse)
print("Train R2:", train_r2)
# print("Validation R2:", val_r2)
print("Test R2:", test_r2)

# Function to plot scatter plot of dataset and prediction line
def plot_scatter_with_prediction(X_test, y_test, model, datasettype):
    # Plot scatter plot of test data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_test[:, 0], y_test, label='Actual', color='blue', alpha=0.5)
    
    # Predict global sales using the model
    predictions = model.predict(X_test)
    
    # Plot prediction line
    ax.scatter(X_test[:, 0], predictions, label='Predicted', color='red', alpha=0.5)
    
    # Set labels for x and y axes
    ax.set_xlabel('Features')
    ax.set_ylabel('Global_Sales')
    ax.set_title('Scatter Plot with Prediction Line on '+ datasettype)
    
    # Add legend
    ax.legend()
    
    plt.show()

# Plot scatter plot with prediction line for test set
end = time.time()
print(end-start)
plot_scatter_with_prediction(X_test, y_test, model, 'Test set')

# Plot scatter plot with prediction line for validation set
# plot_scatter_with_prediction(X_val, y_val, model, 'Validation set')
