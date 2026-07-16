import os
import json
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

# Müştəri rəylərinin analizi üçün JSON çıxış gözləyən sistem promptu
JSON_SYSTEM_PROMPT = """
Sən müştəri rəylərini analiz edən köməkçisən. Sənə gələn rəyi analiz et və mütləq və mütləq aşağıdakı JSON formatında cavab qaytar. 
JSON-dan kənar heç bir izah, giriş və ya yekun mətni yazma.

Gözlənilən format:
{
  "sentiment": "müsbət" və ya "mənfi",
  "reason_az": "analizin qısa izahı"
}
"""

def analyze_review_and_validate(user_review: str):
    """
    Rəyi analiz edən, rəsmi OpenAI API vasitəsilə JSON formatında cavab alan,
    onu təmizləyən və düzgünlüyünü valide edən funksiya.
    """
    messages = [
        {"role": "system", "content": JSON_SYSTEM_PROMPT},
        {"role": "user", "content": user_review}
    ]
    
    try:
        print("[Sistem] OpenAI-a sorğu göndərilir...")
        
        # response_format parametrini əlavə edirik ki, model mütləq JSON qaytarsın
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            timeout=15.0
        )
        
        raw_output = response.choices[0].message.content
        print(f"\n[Sistem] Modelin çiy cavabı (Raw Output):\n{raw_output}\n")
        
        # === VALIDASİYA VƏ PARSING MƏRHƏLƏSİ ===
        # Modeldən gələn mətni təmizləyirik (sağdakı-soldakı boşluqlar və markdown-lar silinir)
        cleaned_output = raw_output.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        elif cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:]
            
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]
            
        cleaned_output = cleaned_output.strip()
        
        # JSON-u parse etməyə çalışırıq
        parsed_data = json.loads(cleaned_output)
        
        # Şema validasiyası: Lazımi açarların JSON daxilində olub-olmadığını yoxlayırıq
        required_keys = ["sentiment", "reason_az"]
        for key in required_keys:
            if key not in parsed_data:
                raise KeyError(f"JSON daxilində tələb olunan '{key}' açarı tapılmadı!")
                
        return parsed_data, True
        
    except json.JSONDecodeError as je:
        print(f"[Xəta] Gələn məlumat düzgün JSON formatında deyil: {je}")
        return None, False
    except Exception as e:
        print(f"[Xəta] Başqa bir problem yarandı: {e}")
        return None, False

if __name__ == "__main__":
    test_review = "Kuryer yeməyi çox gec gətirdi, həm də gələndə hər şey artıq buz kimi soyumuşdu."
    print("Sistem işə salınır...")
    
    parsed_json, is_valid = analyze_review_and_validate(test_review)
    
    if is_valid:
        print("--- [UĞURLU] JSON Uğurla Parse və Valide Olundu! ---")
        print(f"Sentiment: {parsed_json['sentiment']}")
        print(f"İzah: {parsed_json['reason_az']}")
    else:
        print("--- [UĞURSUZ] Validasiya xətası baş verdi! ---")