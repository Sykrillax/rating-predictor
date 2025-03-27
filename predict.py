import pandas as pd
import joblib

# 🔹 1. Load Model Terbaik
model = joblib.load("best_model.pkl")

# 🔹 2. Load Data yang Akan Diprediksi
df = pd.read_csv("TFIDF_Reviews.csv")

# 🔹 3. Pastikan Tidak Ada Nilai Kosong
df = df.dropna()

# 🔹 4. Pisahkan Fitur (X) dan Simpan Data Asli
X = df.drop(columns=["score"])
df_original = df.copy()

# 🔹 5. Prediksi Rating dengan Model
df_original["predicted_score"] = model.predict(X)

# 🔹 6. Konversi Rating Kembali ke Skala [1-5]
df_original["predicted_score"] = df_original["predicted_score"] + 1

# 🔹 7. Simpan Hasil Prediksi ke File CSV
df_original.to_csv("Predicted_Reviews.csv", index=False)

print("✅ Prediksi selesai! Hasil disimpan di 'Predicted_Reviews.csv'")
