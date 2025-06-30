from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ✅ Replace these with real values later if needed
true_ages = [70, 64, 67, 66, 75]
predicted_ages = [70.8, 64.37, 67.09, 66.74, 75.45]

# 📊 Evaluation Metrics
mae = mean_absolute_error(true_ages, predicted_ages)
mse = mean_squared_error(true_ages, predicted_ages)
rmse = mse ** 0.5  # manually calculate RMSE

print(f"📏 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}")

# 📈 Visualization
plt.plot(true_ages, label='True Age', marker='o')
plt.plot(predicted_ages, label='Predicted Age', marker='x')
plt.title("True vs Predicted Age")
plt.xlabel("Sample")
plt.ylabel("Age")
plt.legend()
plt.grid(True)
plt.show()
