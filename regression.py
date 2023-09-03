import numpy as np
import matplotlib.pyplot as plt

# Generate some random data for demonstration
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Independent variable
y = 2 * X + 1 + np.random.randn(100, 1)  # Dependent variable with noise


learning_rate = 0.01
num_iterations = 1000

#weights with zeros
theta0 = 0
theta1 = 0

#gradient descent 
for i in range(num_iterations):
    y_pred = theta0 + theta1 * X
    error = y_pred - y
    
    # Update weights
    theta0 -= learning_rate * np.mean(error)
    theta1 -= learning_rate * np.mean(error * X)

# Make predictions on the data
y_pred = theta0 + theta1 * X

# Calculate the Mean Squared Error (MSE)
mse = np.mean((y_pred - y) ** 2)

print(f"Mean Squared Error: {mse:.2f}")

# Plot the data points and the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red', linewidth=2)
plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (y)")
plt.title("Linear Regression")
plt.show()

# Print the coefficients (weights)
print(f"Theta0 (intercept): {theta0:.2f}")
print(f"Theta1 (slope): {theta1:.2f}")
