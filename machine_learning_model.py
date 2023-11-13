import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import numpy as np

# Start timing the entire process
start_time = time.time()

# Load the training CSV file into a pandas DataFrame
df = pd.read_csv('d2 data/ising_model_data.csv')

# Separate the features and target
X = df.drop(columns=['magnetization'])
y = df['magnetization']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Specify configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Train the model
print("Starting training...")
training_start = time.time()
gbm = lgb.train(params,
                train_data,
                num_boost_round=20,
                valid_sets=train_data,
                )
training_end = time.time()

# Predict on the test set
print("Starting prediction...")
prediction_start = time.time()
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
prediction_end = time.time()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Save the model
gbm.save_model('d2 data/ising_model_lgbm_model.txt')

# Load the validation data
validation_df = pd.read_csv('d2 data/validation_ising_model_data.csv')

# Prepare the feature data for prediction
X_validation = validation_df.drop(columns=['magnetization'])

# Predict magnetization using the loaded model
predicted_magnetization = gbm.predict(X_validation, num_iteration=gbm.best_iteration)

# Evaluate the validation predictions
validation_mse = mean_squared_error(validation_df['magnetization'], predicted_magnetization)
validation_rmse = np.sqrt(validation_mse)

# Calculate and print the statistics
data_preparation_time = 0  # Assuming this was already done prior to the script
model_and_feature_tuning_time = training_end - training_start
prediction_time = prediction_end - prediction_start
total_time = time.time() - start_time

print(f"Data preparation: {data_preparation_time:.2f} seconds")
print(f"Model and feature tuning: {model_and_feature_tuning_time:.2f} seconds")
print(f"Prediction time: {prediction_time:.2f} seconds")
print(f"Total time: {total_time:.2f} seconds")
print(f"Validation score: RMSE = {validation_rmse:.4f}")
