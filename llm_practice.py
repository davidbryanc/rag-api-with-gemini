from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# --- Bagian Utama untuk Menjalankan Skrip ---
if __name__ == "__main__":
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY tidak ditemukan. Pastikan file .env Anda sudah benar.")
        
    # Konfigurasi tetap sama
    client = genai.Client(api_key=api_key)
    print("Klien LLM Gemini berhasil dikonfigurasi dengan pustaka terbaru.")
    
    response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=types.Part.from_text(text='Jelaskan apa itu fotosintesis dalam satu kalimat.'),
    config=types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        top_k=20,
    ))
    
    print(response.text)
