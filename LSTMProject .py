### Frozen Dessert Production Prediction using LSTM

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
# The dataset is assumed to be in a CSV file with a "DATE" column for the index and production values
# Change '../DATA/Frozen_Dessert_Production.csv' to the correct path if needed
df = pd.read_csv('../DATA/Frozen_Dessert_Production.csv', index_col='DATE', parse_dates=True)

# Rename the column for clarity
df.columns = ['Production']

# Plot the dataset to visualize trends
df.plot(figsize=(10, 6), title="Frozen Dessert Production Over Time")
plt.show()

# Check the length of the dataset
len(df)

# Define test size and split dataset into train and test
# Test data will include the last 18 months of data
test = 18
test_index = len(df) - test

# Train and test datasets
train = df.iloc[:test_index]
test = df.iloc[test_index:]

# Normalize data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Fit the scaler on the training data
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# Create sequences for time series modeling
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Parameters for sequence generation
length = 12  # Number of past steps to use
n_feat = 1   # Number of features (only production data)

# Generate sequences for training
generator = TimeseriesGenerator(data=scaled_train, targets=scaled_train, length=length, batch_size=1)

# Build the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Initialize the Sequential model
model = Sequential()

# Add an LSTM layer with 150 neurons
model.add(LSTM(150, input_shape=(length, n_feat)))

# Add a Dense layer with 1 neuron (output)
model.add(Dense(1))

# Compile the model with Mean Squared Error loss and Adam optimizer
model.compile(loss='mse', optimizer='adam')

# Prepare validation data
define validation generator
validation_gen = TimeseriesGenerator(scaled_test, scaled_test, length=length, batch_size=1)

# Define Early Stopping to avoid overfitting
from tensorflow.keras.callbacks import EarlyStopping

early = EarlyStopping(monitor='val_loss', patience=2)

# Train the model
history = model.fit(generator, epochs=20, validation_data=validation_gen, callbacks=[early])

# Plot training and validation loss
loss = pd.DataFrame(model.history.history)
loss.plot(figsize=(10, 6), title="Training and Validation Loss")
plt.show()

# Predict on test data
test_predictions = []

# Initial evaluation batch
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape(1, length, n_feat)

# Predict each time step in the test set
for i in range(len(test)):
    # Generate prediction
    current_pred = model.predict(current_batch)[0]

    # Append prediction to the list
    test_predictions.append(current_pred)

    # Update the current batch with the new prediction
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# Inverse transform predictions to original scale
true_predictions = scaler.inverse_transform(test_predictions)

# Add predictions to the test dataframe
test['Predictions'] = true_predictions

# Plot actual vs predicted values
test.plot(figsize=(10, 6), title="Actual vs Predicted Frozen Dessert Production")
plt.show()

# Calculate root mean squared error (RMSE) for evaluation
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(test['Production'], test['Predictions']))
print(f"Root Mean Squared Error: {rmse}")
