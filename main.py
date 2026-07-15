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

# Prompt Engineering Strukturu

# 1. Sistem Promptu: Rol, vəzifə və qaydalar təyin olunur
SYSTEM_PROMPT = """
Sən bir müştəri rəylərini analiz edən peşəkar süni intellekt köməkçisən.
Sənə daxil olan istifadəçi rəyini oxuyub, mütləq aşağıdakı ciddi JSON formatında cavab verməlisən.
Kənar heç bir giriş və ya izah mətni yazma, sadəcə və sadəcə JSON obyekti qaytar.

Gözlənilən JSON formatı:
{
  "sentiment": "müsbət" və ya "mənfi" və ya "neytral",
  "topic": "məhsul keyfiyyəti" və ya "kuryer/çatdırılma" və ya "müştəri xidməti" və ya "digər",
  "summary_az": "rəyin qısa 1 cümləlik Azərbaycan dilində xülasəsi"
}
"""


# 2. Few-Shot Nümunələri (Modelə necə cavab verəcəyini öyrətmək üçün)
FEW_SHOT_EXAMPLES = [
    # Nümunə 1:
    {"role": "user", "content": "Kuryer yeməyi gətirəndə hər şey dağılmışdı və çox soyuq idi!"},
    {"role": "assistant", "content": '{"sentiment": "mənfi", "topic": "kuryer/çatdırılma", "summary_az": "Çatdırılma zamanı yeməyin dağılması və soyuq olması şikayəti."}'},
    
    # Nümunə 2:
    {"role": "user", "content": "Telefonun kamerası həqiqətən əladır, şəkillər çox keyfiyyətli çıxır."},
    {"role": "assistant", "content": '{"sentiment": "müsbət", "topic": "məhsul keyfiyyəti", "summary_az": "Telefonun kamerasının və foto keyfiyyətinin bəyənilməsi."}'}
]

def analyze_customer_review(review_text: str):
    """
    Sistem, Few-shot və User promptlarını birləşdirib API-a göndərən funksiya.
    """
    # Mesajlar massivinin qurulması
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Few-shot nümunələrinin əlavə edilməsi
    messages.extend(FEW_SHOT_EXAMPLES)
    
    # Real istifadəçi sorğusunun əlavə edilməsi (User Prompt)
    messages.append({"role": "user", "content": review_text})
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1 # Cavabın daha stabil və dəqiq olması üçün temperature-i aşağı saxlayırıq
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Xəta baş verdi: {e}"
    
if __name__ == "__main__":
    # Test etmək üçün fərqli bir rəy veririk
    test_review = "Dəstək komandasındakı xanım mənə çox nəzakətli davrandı və problemimi 5 dəqiqədə həll etdi."
    
    print("Müştəri rəyi analiz edilir...")
    analiz_neticesi = analyze_customer_review(test_review)
    
    print("\n--- Analiz Nəticəsi (JSON) ---")
    print(analiz_neticesi)