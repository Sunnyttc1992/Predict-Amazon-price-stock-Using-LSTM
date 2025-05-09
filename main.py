# import 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timezone
import os


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the dataset
df = pd.read_csv('AMZN_2012-05-19_2025-04-17.csv')
print(df.head())
print(df.info())
print(df.describe())

# Convert 'date' to datetime format early and preserve it
df['date'] = pd.to_datetime(df['date'], utc=True)
date_col = df['date']

# Plot 1: Open and Close prices
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['close'], label='Close Price', color='blue')
plt.plot(df['date'], df['open'], label='Open Price', color='orange')
plt.title('AMZN Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Trading volume
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['volume'], label='Volume', color='green')
plt.title('AMZN Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Keep only numeric columns (but retain date separately)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df = df[numeric_cols]
df['date'] = date_col

# Plot 3: Correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.drop(columns=['date']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Slice the prediction period
prediction_period = df.loc[
    (df['date'] > datetime(2025, 1, 1, tzinfo=timezone.utc)) &
    (df['date'] < datetime(2025, 4, 1, tzinfo=timezone.utc))
]

plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['close'], label='Close Price', color='blue')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("AMZN Stock Price Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Prepare data for LSTM
stock_close = df['close'].values
dataset = stock_close.reshape(-1, 1)

# Normalize the data
scaler = StandardScaler()
dataset_scaled = scaler.fit_transform(dataset)

# Training/test split
training_data_len = int(np.ceil(len(dataset_scaled) * 0.95))
training_data = dataset_scaled[:training_data_len]

# Create sliding windows
X_train, y_train = [], []
for i in range(60, len(training_data)):
    X_train.append(training_data[i - 60:i, 0])
    y_train.append(training_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Build the LSTM model
model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])

model.summary()

# Compile and train
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
history = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1)

# Prepare test data
test_data = dataset_scaled[training_data_len - 60:, :]
X_test, y_test = [], dataset_scaled[training_data_len:, :]

for i in range(60, len(test_data)):
    X_test.append(test_data[i - 60:i, 0])

X_test = np.array(X_test).reshape(-1, 60, 1)

# Predict
predictions = model.predict(X_test)
predictions_unscaled = scaler.inverse_transform(predictions)
y_test_unscaled = scaler.inverse_transform(y_test)

# Align date range
test_dates = df['date'].values[training_data_len:]

# Create DataFrame for plotting
test_df = pd.DataFrame({
    'date': test_dates,
    'close': y_test_unscaled.flatten(),
    'Predictions': predictions_unscaled.flatten()
})

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(df['date'][:training_data_len], dataset[:training_data_len], label='Train', color='blue')
plt.plot(test_df['date'], test_df['close'], label='Test', color='orange')
plt.plot(test_df['date'], test_df['Predictions'], label='Predictions', color='red')
plt.title('Stock Price Prediction with LSTM')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Evaluate performance
mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
r2 = r2_score(y_test_unscaled, predictions_unscaled)

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R² Score: {r2:.4f}")

actual_dir = np.sign(np.diff(y_test_unscaled.flatten()))
pred_dir = np.sign(np.diff(predictions_unscaled.flatten()))
direction_acc = np.mean(actual_dir == pred_dir)
print(f"Direction Accuracy: {direction_acc:.2%}")

print(f"Predictions: {predictions_unscaled.flatten()}")
print(f"Actual: {y_test_unscaled.flatten()}")
