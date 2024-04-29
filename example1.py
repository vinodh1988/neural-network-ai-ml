import numpy as np
import tensorflow as tf
from tensorflow import keras

# Generate synthetic data: y = 2x + 1 with some noise
np.random.seed(0)
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 2 * X + 1 + np.random.normal(0, 0.1, (200,))

# Split data into training and testing
X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

# Build the model
model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=(1,))
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=50, batch_size=10)

# Evaluate the model
loss = model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss}')

# Predict new values
new_X = np.array([0.5, -0.5, 0.2])
predictions = model.predict(new_X)
print(f'Predictions: {predictions.flatten()}')
