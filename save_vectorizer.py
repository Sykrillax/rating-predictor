import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 🔹 Load Data Preprocessed Reviews
df = pd.read_csv("Cleaned_Reviews.csv")

# 🔹 Pastikan kolom 'clean_content' ada
if "clean_content" in df.columns:
    df["clean_content"] = df["clean_content"].fillna("")

    # 🔹 Inisialisasi TF-IDF Vectorizer dengan max_features lebih besar & bigram
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df["clean_content"])

    # 🔹 Simpan hasil TF-IDF ke CSV
    df_tfidf = pd.DataFrame(X_tfidf.toarray())
    df_tfidf["score"] = df["score"]
    df_tfidf.to_csv("TFIDF_Reviews.csv", index=False)

    # 🔹 Simpan vectorizer ke file
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ TF-IDF Vectorizer berhasil disimpan!")
else:
    print("⚠️ Kolom 'clean_content' tidak ditemukan dalam dataset!")
