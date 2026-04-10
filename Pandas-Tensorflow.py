import pandas as pd
import tensorflow as tf
df = pd.read_csv("fortune1000.csv")

# Select only numerical features for X to make it compatible with a simple Dense layer
# Assuming 'Rank', 'Revenue', 'Profits' are the numerical columns other than 'Employees'
numerical_features = ['Rank', 'Revenue', 'Profits']
X = df[numerical_features].values
y = df['Employees'].values

# Define the model before using it
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(1) #Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

model.fit(X,y, epochs=10)
print(model)
