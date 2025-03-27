import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings

# Konfigurasi
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# Load dataset
df = pd.read_csv("Cleaned_Reviews.csv")

# Rating Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=df['score'], palette="viridis")
plt.title("Distribusi Rating Produk")
plt.xlabel("Rating (1-5)")
plt.ylabel("Jumlah Review")
plt.show()

# Text Length
df['text_length'] = df['clean_content'].astype(str).apply(len)

plt.figure(figsize=(8, 5))
sns.histplot(df['text_length'], bins=30, kde=True, color="purple")
plt.title("Distribusi Panjang Teks Review")
plt.xlabel("Jumlah Karakter dalam Review")
plt.ylabel("Frekuensi")
plt.show()

# Word Cloud 
text = " ".join(df['clean_content'].dropna())

wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud dari Kata-Kata dalam Review")
plt.show()

print("✅ EDA selesai! Visualisasi telah ditampilkan.")
