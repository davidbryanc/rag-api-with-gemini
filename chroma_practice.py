# Siapkan "dokumen" atau basis pengetahuan kita.
# Dalam aplikasi nyata, ini bisa berasal dari file PDF, database, dll.
# Setiap item dalam list ini akan menjadi satu 'dokumen' di ChromaDB.
documents = [
    "Tokopedia, didirikan pada tahun 2009, adalah salah satu marketplace terbesar di Indonesia.",
    "Go-Jek, sekarang GoTo, memulai layanannya sebagai call center ojek pada tahun 2010.",
    "Pada tahun 2021, Go-Jek dan Tokopedia melakukan merger untuk membentuk GoTo Group.",
    "Traveloka, didirikan tahun 2012, adalah platform pemesanan tiket pesawat dan hotel online.",
    "Bukalapak merupakan salah satu pesaing utama Tokopedia di pasar e-commerce Indonesia.",
    "Soekarno adalah presiden pertama Republik Indonesia, memproklamasikan kemerdekaan pada tahun 1945.",
    "Rendang, masakan daging pedas dari Minangkabau, sering disebut sebagai salah satu makanan terlezat di dunia."
]

# Kita juga perlu ID unik untuk setiap dokumen.
# Ini penting untuk memperbarui atau menghapus dokumen nanti.
document_ids = [
    "doc1", 
    "doc2", 
    "doc3", 
    "doc4", 
    "doc5", 
    "doc6",
    "doc7"
]

import chromadb

# Buat klien ChromaDB. 
# PersistentClient akan menyimpan data ke disk di folder yang ditentukan.
# Jika folder belum ada, ia akan membuatnya.
client = chromadb.PersistentClient(path="./chroma_db")

# Buat koleksi baru atau muat yang sudah ada.
# Jika koleksi dengan nama 'indonesian_tech' sudah ada, ia akan menggunakannya.
# Jika belum, ia akan membuatnya.
collection_name = "indonesian_tech"
print(f"Membuat atau memuat koleksi: {collection_name}")
collection = client.get_or_create_collection(name=collection_name)

# 4. Tambahkan dokumen ke koleksi.
#    Kita memberikan daftar dokumen dan ID-nya.
#    ChromaDB akan secara otomatis:
#    a. Menggunakan model embedding default (all-MiniLM-L6-v2) untuk mengubah setiap dokumen menjadi vektor.
#    b. Menyimpan pasangan (dokumen, vektor, id) ke dalam database.
print("Menambahkan dokumen ke koleksi...")
collection.add(
    documents=documents,
    ids=document_ids
)
print(f"Berhasil menambahkan {collection.count()} dokumen.")


# 5. Lakukan pencarian (query).
print("\n" + "="*50)
print("             MELAKUKAN PENCARIAN SEMANTIK")
print("="*50)

# Pertanyaan yang ingin kita ajukan
query_text = "Kapan Go-Jek memulai layanannya?"

# Lakukan query ke koleksi.
# ChromaDB akan:
# a. Mengubah `query_text` menjadi sebuah vektor menggunakan model yang sama.
# b. Mencari di dalam database untuk N vektor yang paling mirip.
# c. Mengembalikan dokumen-dokumen yang terkait dengan vektor-vektor tersebut.
print(f"Pertanyaan: '{query_text}'")
print("\nHasil pencarian relevan:")
results = collection.query(
    query_texts=[query_text],
    n_results=2 # Minta 2 hasil teratas
)

# Cetak hasilnya dengan rapi
for i, doc in enumerate(results['documents'][0]):
    # 'distances' adalah ukuran seberapa "jauh" hasil dari query. Semakin kecil semakin baik.
    distance = results['distances'][0][i]
    print(f"  {i+1}. Teks: '{doc}'")
    print(f"     Jarak: {distance:.4f}\n")

# --- Eksperimen Kedua ---
print("-" * 50)
query_text_2 = "Siapa pemimpin pertama negara Indonesia?"
print(f"Pertanyaan: '{query_text_2}'")
print("\nHasil pencarian relevan:")
results_2 = collection.query(
    query_texts=[query_text_2],
    n_results=2
)
for i, doc in enumerate(results_2['documents'][0]):
    distance = results_2['distances'][0][i]
    print(f"  {i+1}. Teks: '{doc}'")
    print(f"     Jarak: {distance:.4f}\n")
