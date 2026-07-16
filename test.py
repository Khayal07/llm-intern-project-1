import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL")
MODEL_NAME = os.getenv("LLM_MODEL_NAME")

if not API_KEY:
    raise ValueError("XƏTA: .env faylında API açarı tapılmadı!")


client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=API_KEY
)

SYSTEM_PROMPT = """
Sən köməkçi bir süni intellektsən. Sənə verilən suallara maraqlı və dolğun cavablar verməlisən.
"""

def analyze_with_streaming(user_prompt: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True
        )
        
        print("[Sistem] Model real vaxt rejiminde cavab verir: \n")
        print("Cavab: ", end="", flush=True)
        
        for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices[0].delta else ""
            if delta:
                print(delta, end="", flush=True)
                
        print("\n\n[Sistem] Axın uğurla başa çatdı.")
        
    except Exception as e:
        print(f"\nXəta baş verdi: {e}")
        
        
if __name__ == "__main__":
    test_question = "Süni intellektin gələcəyi haqqında 2 cümləlik qısa fəlsəfi fikir yaz."
    analyze_with_streaming(test_question)