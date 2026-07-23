import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# === API KONFİQURASİYASI (Checkpoint 1 - API inteqrasiyası) ===
# Layihə API provayderi olaraq OpenRouter platformasından istifadə edir.
# Açar və endpoint (base_url) ətraf mühit dəyişənləri vasitəsilə təhlükəsiz oxunur.
API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# LLM_MODEL_NAME yoxlanılır; dəyər verilmədikdə təhlükəsiz standart model seçilir
MODEL_NAME = os.getenv("LLM_MODEL_NAME") or "openai/gpt-4o-mini"

if not API_KEY:
    raise ValueError(
        "XƏTA: .env faylında OPENROUTER_API_KEY (və ya OPENAI_API_KEY) tapılmadı!"
    )

# OpenAI SDK OpenRouter endpoint-inə (base_url) yönləndirilir
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# === MODEL TARİFLƏRİ (Checkpoint 6 - 1 Milyon Token üçün USD ilə) ===
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o": {"input": 2.500, "output": 10.000},
    "default": {"input": 0.150, "output": 0.600}  # Model tapılmadıqda tətbiq olunan standart tarif
}

def calculate_cost(prompt_tokens: int, completion_tokens: int, model_name: str):
    """
    İstifadə olunan token miqdarına və modelə uyğun olaraq sorğunun real xərcini hesablayır.
    """
    # OpenRouter model adları "provider/model" formatındadır; tarif üçün yalnız model adını götürürük
    normalized_name = (model_name or "").split("/")[-1]
    pricing = MODEL_PRICING.get(normalized_name, MODEL_PRICING["default"])
    
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost

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
    onun token/xərc monitorinqini aparan, təmizləyən və düzgünlüyünü valide edən funksiya.
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
        
        # === MONITORİNQ VƏ TOKEN HESABLANMASI (Checkpoint 6) ===
        usage = response.usage
        if usage:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            cost = calculate_cost(prompt_tokens, completion_tokens, MODEL_NAME)
            
            print("--- [MONİTORİNQ: Token və Xərc] ---")
            print(f"İstifadə edilən Model: {MODEL_NAME}")
            print(f"Giriş (Prompt) Tokenləri: {prompt_tokens}")
            print(f"Çıxış (Completion) Tokenləri: {completion_tokens}")
            print(f"Ümumi Token Sayı: {total_tokens}")
            print(f"Sorğunun Təxmini Xərci: ${cost:.6f} USD")
            print("-----------------------------------\n")
        else:
            print("[Sistem/Xəbərdarlıq] Modelin istifadə (usage) məlumatları alınmadı.\n")

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