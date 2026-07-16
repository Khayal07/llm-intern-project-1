import os
import time 
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL_NAME")

if not API_KEY:
    raise ValueError("XƏTA: .env faylında OPENAI_API_KEY tapılmadı!")

client = OpenAI(
    api_key=API_KEY
)

def run_llm_task_with_retry(user_prompt: str, max_retries: int = 3, initial_delay: int = 2):
    """
    Sorğunu göndərən və xəta baş verdikdə Exponential Backoff
    məntiqi ilə yenidən cəhd edən (Retry) funksiya.
    """
    messages = [
        {"role": "system", "content": "Sən köməkçi bir botsan."},
        {"role": "user", "content": user_prompt}
    ]
    
    delay = initial_delay
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Sistem] Sorğu göndərilir... (Cəhd {attempt}/{max_retries})")
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                timeout=10.0 # Cavab gecikərsə gözləmə limiti
            )
            
            # Əgər uğurlu olarsa cavabı dərhal qaytarırıq
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[Sistem Xətası] Qoşulma zamanı xəta baş verdi: {e}")
            
            if attempt < max_retries:
                print(f"[Retry] {delay} saniyə gözlənilir və yenidən cəhd edilir...")
                time.sleep(delay) 
                delay *= 2  # Gözləmə müddətini eksponensial olaraq artırırıq (2s -> 4s -> 8s)
            else:
                print("[Sistem] Bütün cəhdlər uğursuz oldu.")
                return f"Xəta: API-a qoşulmaq mümkün olmadı. Detal: {e}"

if __name__ == "__main__":
    test_prompt = "Azərbaycanın ən böyük gölü hansıdır?"
    print("Sistem işə salınır...")
    
    cavab = run_llm_task_with_retry(test_prompt)
    print("\n--- Modelin Cavabı ---")
    print(cavab)