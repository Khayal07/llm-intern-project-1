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


def run_checkpoint1(prompt: str):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Qosulma xetasi: {e}"
    
if __name__ == "__main__":
    test_prompt = "Hello ai, how are u?"
    cavab = run_checkpoint1(test_prompt)
    print(cavab)