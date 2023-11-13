import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assuming 'ising_model_data.csv' has been generated by the simulation code you provided.
# Load the CSV file into a pandas DataFrame
df = pd.read_csv('d2 data/ising_model_data.csv')

# Separate the features and target
X = df.drop(columns=['magnetization'])  # Features including state and temperature
y = df['magnetization']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Specify your configurations as a dict
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
gbm = lgb.train(params,
                train_data,
                num_boost_round=20,
                valid_sets=train_data,  # Validate with the same training data
                )

# Predict on the test set
print("Starting prediction...")
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'The mean squared error of prediction is: {mse}')

# Save the model
gbm.save_model('d2 data/ising_model_lgbm_model.txt')

# Output: Training the model, making predictions, and evaluating the model

# Run the validation

# Load the LightGBM model
model = lgb.Booster(model_file='d2 data/ising_model_lgbm_model.txt')

# Load the validation data
validation_df = pd.read_csv('d2 data/validation_ising_model_data.csv')

# Prepare the feature data for prediction
X_validation = validation_df.drop(columns=['magnetization'])  # assuming you have a 'magnetization' column in validation data

# Predict magnetization using the loaded model
predicted_magnetization = model.predict(X_validation, num_iteration=model.best_iteration)

# If you want to add the predictions as a new column to the validation dataframe
validation_df['predicted_magnetization'] = predicted_magnetization

# Save the updated dataframe to a new CSV file if needed
validation_df.to_csv('d2 data/validation_ising_model_with_predictions.csv', index=False)

print("Predictions added to the validation dataset and saved.")