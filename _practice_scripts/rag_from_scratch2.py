# rag_from_scratch.py

import os
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- BAGIAN 1: SETUP SEMUA KOMPONEN ---

def setup_rag_components():
    """
    Menginisialisasi dan mengonfigurasi semua komponen yang dibutuhkan untuk RAG.
    Mengembalikan klien ChromaDB dan koleksi yang sudah terisi.
    """
    # Konfigurasi Database Vektor (ChromaDB)
    # Kita menggunakan kembali database yang sudah dibuat di Hari ke-10
    client_db = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "indonesian_tech"
    collection = client_db.get_collection(name=collection_name)
    print(f"✅ Komponen Database Vektor (ChromaDB) berhasil dimuat. Koleksi '{collection_name}' memiliki {collection.count()} dokumen.")
    
    return collection

# Inisialisasi komponen saat skrip dimulai
try:
    chroma_collection = setup_rag_components()
except Exception as e:
    print(f"Gagal melakukan setup: {e}")
    # Jika gagal, mungkin karena data belum diindeks.
    # Anda bisa menyarankan pengguna untuk menjalankan skrip Hari ke-10.
    print("Pastikan Anda sudah menjalankan skrip 'chroma_practice.py' dari Hari ke-10 terlebih dahulu untuk membuat database.")
    exit()

# ... (kode setup sebelumnya)

def ask_gemini(prompt: str) -> str:
    """
    Mengirimkan prompt ke model Gemini dan mengembalikan respons teks.
    (Sama seperti kemarin, tapi tanpa blok __main__)
    """
    try:
            # Konfigurasi LLM Gemini
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY tidak ditemukan.")
        client = genai.Client(api_key=api_key)
        print("✅ Komponen LLM (Gemini) berhasil dikonfigurasi.")
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=types.Part.from_text(text=prompt),
            config=types.GenerateContentConfig(
                temperature=0,
                top_p=0.95,
                top_k=20,
        ))
        
        return response.text
    except Exception as e:
        return f"Error saat memanggil API Gemini: {e}"

def answer_question(query: str, collection):
    """
    Fungsi utama RAG: Mengambil konteks, membuat prompt, dan menghasilkan jawaban.
    """
    print(f"\n🔎 Menerima pertanyaan: '{query}'")

    # --- LANGKAH 1: RETRIEVE ---
    # Mencari dokumen relevan dari ChromaDB.
    # Kita mengambil 3 hasil untuk konteks yang lebih kaya.
    results = collection.query(
        query_texts=[query],
        n_results=3 
    )
    retrieved_documents = results['documents'][0]
    print("📚 Konteks yang berhasil diambil dari database:")
    for i, doc in enumerate(retrieved_documents):
        print(f"  {i+1}. {doc}")

    # --- LANGKAH 2: AUGMENT ---
    # Menggabungkan dokumen-dokumen yang diambil menjadi satu string konteks.
    context_string = "\n\n".join(retrieved_documents)

    # --- LANGKAH 3: GENERATE ---
    # Membuat prompt akhir dengan menggunakan template.
    prompt_template = f"""
Anda adalah asisten AI yang cerdas dan faktual. Gunakan HANYA informasi dari 'KONTEKS' yang diberikan di bawah ini untuk menjawab 'PERTANYAAN'.
Jangan menggunakan pengetahuan eksternal Anda.
Jika jawaban tidak dapat ditemukan di dalam 'KONTEKS', jawab dengan 'Maaf, saya tidak dapat menemukan informasi mengenai hal tersebut di dalam data saya.'

KONTEKS:
{context_string}

PERTANYAAN:
{query}

JAWABAN:
"""
    print("\n📝 Membuat prompt akhir untuk LLM...")
    # print(prompt_template) # Anda bisa uncomment ini untuk melihat prompt lengkapnya

    # Memanggil LLM dengan prompt yang sudah diperkaya
    print("🧠 Mengirim prompt ke Gemini untuk menghasilkan jawaban...")
    answer = ask_gemini(prompt_template)
    
    return answer

# ... (kode fungsi sebelumnya)

# --- BAGIAN UTAMA UNTUK PENGUJIAN ---
if __name__ == "__main__":
    
    # --- UJI KASUS 1: Pertanyaan yang Jawabannya ADA di Database ---
    print("\n" + "="*50)
    print("             UJI KASUS 1: INFORMASI TERSEDIA")
    print("="*50)
    pertanyaan_1 = "Kapan Tokopedia dan Go-Jek bergabung?"
    jawaban_1 = answer_question(pertanyaan_1, chroma_collection)
    print("\n✅ Jawaban Akhir dari Sistem RAG:")
    print(jawaban_1)
    print("="*50)

    # --- UJI KASUS 2: Pertanyaan yang Jawabannya TIDAK ADA di Database ---
    print("\n" + "="*50)
    print("             UJI KASUS 2: INFORMASI TIDAK TERSEDIA")
    print("="*50)
    pertanyaan_2 = "Apa masakan favorit Soekarno?"
    jawaban_2 = answer_question(pertanyaan_2, chroma_collection)
    print("\n✅ Jawaban Akhir dari Sistem RAG:")
    print(jawaban_2)
    print("="*50)
