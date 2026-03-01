# langchain_rag.py

import os
from dotenv import load_dotenv

# Impor komponen-komponen LangChain
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. SETUP LINGKUNGAN ---
# Sama seperti kemarin, muat kunci API
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY tidak ditemukan. Pastikan file .env Anda sudah benar.")

print("✅ Lingkungan siap.")

# --- 2. INISIALISASI KOMPONEN ---

# Inisialisasi model LLM (Gemini)
# Kita bisa langsung mengatur temperature di sini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0)
print("✅ Model LLM (Gemini) diinisialisasi.")

# Inisialisasi model embedding
# Kita harus secara eksplisit memberitahu LangChain model apa yang harus digunakan
# agar konsisten dengan database yang kita buat di Hari ke-10.
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Hubungkan ke ChromaDB yang sudah ada dan buat retriever
# Retriever ini adalah komponen yang akan melakukan pencarian dokumen.
vector_store = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embedding_function,
    collection_name="indonesian_tech"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # k adalah sama dengan n_results
print(f"✅ Retriever dari ChromaDB ('{vector_store._collection.name}') siap.")

# Definisikan template prompt kita
prompt_template_str = """
Anda adalah asisten AI yang cerdas dan faktual. Gunakan HANYA informasi dari 'KONTEKS' yang diberikan di bawah ini untuk menjawab 'PERTANYAAN'.
Jangan menggunakan pengetahuan eksternal Anda.
Jika jawaban tidak dapat ditemukan di dalam 'KONTEKS', jawab dengan 'Maaf, saya tidak dapat menemukan informasi mengenai hal tersebut di dalam data saya.'

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:
"""
prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
print("✅ Template prompt dibuat.")

# --- 3. BANGUN RANTAI RAG MENGGUNAKAN LCEL ---

# Ini adalah inti dari LangChain. Kita mendeklarasikan aliran data.
print("⛓️ Membangun rantai RAG...")
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
print("✅ Rantai RAG berhasil dibangun.")

# --- 4. PANGGIL RANTAI UNTUK MENJAWAB PERTANYAAN ---

if __name__ == "__main__":
    # --- UJI KASUS 1: Pertanyaan yang Jawabannya ADA ---
    print("\n" + "="*50)
    print("             UJI KASUS 1: INFORMASI TERSEDIA")
    print("="*50)
    
    question_1 = "Kapan Tokopedia dan Go-Jek bergabung?"
    print(f"Pertanyaan: {question_1}")
    
    # Memanggil rantai sangat mudah dengan .invoke()
    answer_1 = rag_chain.invoke(question_1)
    
    print("\nJawaban dari Rantai LangChain:")
    print(answer_1)
    print("="*50)

    # --- UJI KASUS 2: Pertanyaan yang Jawabannya TIDAK ADA ---
    print("\n" + "="*50)
    print("             UJI KASUS 2: INFORMASI TIDAK TERSEDIA")
    print("="*50)

    question_2 = "Apa masakan favorit Soekarno?"
    print(f"Pertanyaan: {question_2}")
    
    answer_2 = rag_chain.invoke(question_2)
    
    print("\nJawaban dari Rantai LangChain:")
    print(answer_2)
    print("="*50)