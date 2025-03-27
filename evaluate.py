import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 🔹 1. Load Data Asli dan Hasil Prediksi
df = pd.read_csv("Predicted_Reviews.csv")

# 🔹 2. Hitung Metrik Evaluasi
mae = mean_absolute_error(df["score"], df["predicted_score"])
rmse = mean_squared_error(df["score"], df["predicted_score"]) ** 0.5  # Perbaikan RMSE

print(f"📊 Evaluasi Model:")
print(f"➡️ Mean Absolute Error (MAE): {mae:.4f}")
print(f"➡️ Root Mean Squared Error (RMSE): {rmse:.4f}")

# 🔹 3. Visualisasi Distribusi Rating Asli vs Prediksi
plt.figure(figsize=(10, 5))
sns.histplot(df["score"], color="blue", label="Actual", kde=True, bins=5)
sns.histplot(df["predicted_score"], color="red", label="Predicted", kde=True, bins=5)
plt.legend()
plt.title("Distribusi Rating Asli vs Prediksi")
plt.xlabel("Rating")
plt.ylabel("Frekuensi")
plt.show()
