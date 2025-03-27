import pickle
import joblib
from flask import Flask, request, jsonify
import numpy as np

# 🔹 Load TF-IDF Vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# 🔹 Load Model Terbaik
best_model = joblib.load("best_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "API untuk Prediksi Rating Produk 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        review_text = data.get("review")

        if not review_text or not isinstance(review_text, str):
            return jsonify({"error": "Review harus berupa teks dan tidak boleh kosong"}), 400

        # 🔹 Transform review ke bentuk TF-IDF
        review_tfidf = vectorizer.transform([review_text])

        # 🔹 Prediksi rating menggunakan model
        predicted_rating = best_model.predict(review_tfidf)[0] + 1

        return jsonify({"predicted_rating": int(predicted_rating)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
