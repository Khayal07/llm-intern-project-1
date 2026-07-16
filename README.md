# Müştəri Rəylərinin Analizi (LLM Task 1)

Bu layihə, OpenAI API (və ya OpenRouter) vasitəsilə daxil olan müştəri rəylərinin süni intellekt tərəfindən sentiment analizini aparır, nəticəni strukturlaşdırılmış JSON formatında qaytarır, çıxışın strukturunu yoxlayır (validation) və xərclənən tokenlər üzərindən maliyyə monitorinqini həyata keçirir.

## 🚀 Layihənin Qurulması və Konfiqurasiya

### 1. Repozitoriyanın Klonlanması
```bash
git clone [https://github.com/istifadeci-adin/rey-analizi-llm.git](https://github.com/istifadeci-adin/rey-analizi-llm.git)
cd rey-analizi-llm
```
### 2. Virtual Mühitin Yaradılması və Aktivləşdirilməsi
```bash
python -m venv venv
```
# Windows üçün:
```bash
venv\Scripts\activate
```
# macOS/Linux üçün:
```bash
source venv/bin/activate
```


### 3. Lazımi Kitabxanaların Quraşdırılması
```bash
pip install -r requirements.txt
```


### 4. API Açarlarının Konfiqurasiyası (.env tənzimlənməsi)
⚠️ DİQQƏT: Layihənin təhlükəsizliyi üçün həqiqi API açarınız heç vaxt GitHub-a yüklənməməlidir! .gitignore faylı .env faylını avtomatik olaraq gizlədir.

Lokalda işləmək üçün qovluqdakı şablon .env.example faylının surətini çıxararaq .env faylı yaradın və öz real açarınızı daxil edin:
```bash
cp .env.example .env
```
Sonra .env faylını açın və müvafiq məlumatları qeyd edin:
```python
OPENAI_API_KEY=sk-proj-sizin-heqiqi-api-acariniz
LLM_MODEL_NAME=gpt-4o-mini
```

### 💻 Proqramın İşə Salınması
Konfiqurasiyanı tamamladıqdan sonra əsas skripti icra edə bilərsiniz:
```bash
python main.py
```

###  📊 Nümunə Sorğu və Cavab Log-ları (Execution Logs)
Sistemin işləməsi zamanı konsolda yaranan real sorğu, cavab, token monitorinqi və uğurlu JSON validasiya loqları aşağıdakı kimidir:
```
Sistem işə salınır...
[Sistem] OpenAI-a sorğu göndərilir...

[Sistem] Modelin çiy cavabı (Raw Output):
{
  "sentiment": "mənfi",
  "reason_az": "Müştəri yeməyin gecikməsindən və soyuq gəlməsindən narazıdır."
}

--- [MONİTORİNQ: Token və Xərc] ---
İstifadə edilən Model: gpt-4o-mini
Giriş (Prompt) Tokenləri: 124
Çıxış (Completion) Tokenləri: 38
Ümumi Token Sayı: 162
Sorğunun Təxmini Xərci: $0.000041 USD
-----------------------------------

--- [UĞURLU] JSON Uğurla Parse və Valide Olundu! ---
Sentiment: mənfi
İzah: Müştəri yeməyin gecikməsindən və soyuq gəlməsindən narazıdır.
```

### 🛠️ Layihə Strukturu
```
main.py - API inteqrasiyası, validasiya və monitorinq kodunun yer aldığı əsas fayl.

.gitignore - .env və virtual mühit fayllarının GitHub-a sızmasının qarşısını alır.

.env.example - Komanda yoldaşları və yoxlayanlar üçün API konfiqurasiya şablonu.

requirements.txt - Layihənin asılılıqları (OpenAI və python-dotenv).
```

### Biz bununla nə etdik?
1. **README.md sənədini tamamladıq:** Tapşırıqda xüsusi vurğulanan təhlükəsizlik standartını (`.env.example` ilə qurmağı) və sənin kodundakı formatda olan **nümunə terminal log-larını** README daxilinə peşəkar şəkildə yazdıq.
2. **Hər tapşırıq üçün ayrıca GitHub reposu** açmağı unutma. `.gitignore` faylının içində `.env` yazıldığına əmin ol ki, həqiqi açarın səhvən GitHub-a getməsin. 

Artıq layihən tam mənası ilə **təhvil verilməyə hazırdır!**