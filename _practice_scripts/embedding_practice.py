# embedding_practice.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Muat model
print("Memuat model embedding...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model berhasil dimuat.")

# 2. Definisikan kalimat untuk eksperimen utama
print("\nMenyiapkan kalimat untuk eksperimen...")
query = "Mobil itu melaju cepat di jalan tol"
kalimat_mirip = "Kendaraan itu ngebut di jalan bebas hambatan"
kalimat_berbeda = "Saya suka makan kue cokelat"

# Gabungkan semua kalimat untuk di-encode sekaligus agar efisien
all_sentences = [query, kalimat_mirip, kalimat_berbeda]

# 3. Buat embeddings
print("Membuat embeddings...")
embeddings = model.encode(all_sentences)

# Pisahkan embeddings untuk analisis
embedding_query = embeddings[0]
embedding_mirip = embeddings[1]
embedding_berbeda = embeddings[2]

# 4. Hitung kemiripan
print("Menghitung cosine similarity...")
kemiripan_1 = cosine_similarity([embedding_query], [embedding_mirip])
kemiripan_2 = cosine_similarity([embedding_query], [embedding_berbeda])

# 5. Tampilkan hasil untuk pembuktian
print("\n" + "="*50)
print("             HASIL UJI KEMIRIPAN MAKNA")
print("="*50)
print(f"Kalimat Referensi: '{query}'")
print("-" * 50)
print(f"Dibandingkan dengan: '{kalimat_mirip}'")
print(f"Skor Kemiripan: {kemiripan_1[0][0]:.4f}")
print("-" * 50)
print(f"Dibandingkan dengan: '{kalimat_berbeda}'")
print(f"Skor Kemiripan: {kemiripan_2[0][0]:.4f}")
print("="*50)

# Verifikasi parameter sukses
if kemiripan_1[0][0] > kemiripan_2[0][0]:
    print("\n✅ Parameter Sukses Tercapai!")
    print("   Terbukti secara kuantitatif bahwa kalimat tentang mobil lebih mirip satu sama lain.")
else:
    print("\n❌ Parameter Sukses Gagal.")