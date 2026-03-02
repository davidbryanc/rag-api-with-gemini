# app/routers/rag.py

import os
from fastapi import APIRouter, Depends
from dotenv import load_dotenv

# Impor komponen-komponen LangChain & model Pydantic kita
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from ..models.rag_models import QueryRequest, QueryResponse

from fastapi.responses import StreamingResponse
import asyncio

# --- Setup Awal ---
# Kita pindahkan logika setup ke sini.
# Dalam aplikasi produksi, ini sering dilakukan di tempat lain (misal, startup event).

# Muat environment variable
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Inisialisasi komponen-komponen yang mahal hanya sekali
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embedding_function,
    collection_name="indonesian_tech"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Definisikan template prompt
prompt_template_str = """
Anda adalah asisten AI yang cerdas. Gunakan HANYA 'KONTEKS' yang diberikan untuk menjawab 'PERTANYAAN'.
Jika jawaban tidak ada dalam konteks, jawab: 'Maaf, saya tidak dapat menemukan informasi mengenai hal tersebut di dalam data saya.'

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:
"""
prompt_template = ChatPromptTemplate.from_template(prompt_template_str)

# Bangun rantai RAG
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
print("✅ Rantai RAG siap digunakan oleh API.")


# --- Router API ---
router = APIRouter()

@router.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest) -> QueryResponse:
    """
    Menerima pertanyaan, memprosesnya melalui rantai RAG,
    dan mengembalikan jawabannya.
    """
    # Panggil rantai LangChain dengan pertanyaan dari request
    answer_text = rag_chain.invoke(request.question)
    
    # Kembalikan jawaban dalam format Pydantic model QueryResponse
    return QueryResponse(answer=answer_text)

# Buat endpoint baru untuk streaming
@router.post("/ask-stream")
async def ask_question_stream(request: QueryRequest):
    """
    Menerima pertanyaan dan mengembalikan jawabannya sebagai aliran (stream) teks.
    """
    # Definisikan sebuah generator asinkron
    async def event_generator():
        # Gunakan .astream() untuk mendapatkan generator asinkron
        async for chunk in rag_chain.astream(request.question):
            # yield setiap potongan teks saat tiba
            yield chunk
            # Jeda kecil agar streaming terlihat lebih mulus (opsional)
            await asyncio.sleep(0.01)

    # Kembalikan generator di dalam StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/plain")