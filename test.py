import pickle
from sklearn.ensemble import RandomForestRegressor  # Sesuai model yang digunakan

# Pastikan Anda sudah memiliki model yang telah dilatih
best_model = RandomForestRegressor()  # Ganti dengan model hasil training yang benar
best_model.fit(X_train, y_train)  # Pastikan X_train dan y_train sudah didefinisikan

# Simpan model ke dalam file
with open("best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("✅ Model berhasil disimpan!")
