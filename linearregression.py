import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
heights = np.array([150, 160, 170, 180, 190])  # Heights in centimeters
weights = np.array([50, 60, 65, 75, 80])       # Weights in kilograms

# Reshape the data
heights = heights.reshape(-1, 1)
weights = weights.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(heights, weights)

# Make predictions
predicted_weights = model.predict(heights)

# Plotting the results
plt.scatter(heights, weights, color='blue', label='Actual weights')
plt.plot(heights, predicted_weights, color='red', label='Predicted weights (Regression line)')
plt.title('Height vs Weight Prediction using Linear Regression')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend()
plt.show()

# Display the coefficients
print("Slope (beta_1):", model.coef_[0][0])
print("Intercept (beta_0):", model.intercept_[0])
