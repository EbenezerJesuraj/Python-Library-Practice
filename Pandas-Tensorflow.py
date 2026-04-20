import pandas as pd
import tensorflow as tf
import numpy as np

df = pd.read_csv("fortune1000.csv")

# Select only numerical features for X to make it compatible with a simple Dense layer
# Assuming 'Rank', 'Revenue', 'Profits' are the numerical columns other than 'Employees'
numerical_features = ['Rank', 'Revenue', 'Profits']
X = df[numerical_features].values
y = df['Employees'].values

# Define the model before using it
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1) #Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

model.fit(X,y, epochs=10)
model.summary()

# Rank 10, Revenue 50000, Profit 5000
test_company = np.array([[10, 50000, 5000]]),
prediction = model.predict(test_company)

print(f"Predicted Employees: {prediction[0][0]}")
