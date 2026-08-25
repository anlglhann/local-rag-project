# Yerel RAG Soru-Cevap Uygulaması

Bu proje, proje kökündeki `documents.txt` dosyasından ilgili bilgiyi bulup Microsoft Foundry Local üzerinde çalışan küçük bir sohbet modeline veren, sade bir yerel RAG uygulamasıdır. PDF, OCR, bulut API’si veya kullanıcı hesabı içermez.

## Nasıl çalışır?

1. `documents.txt`, paragraf ve cümle sınırları korunarak küçük parçalara ayrılır.
2. Parçalar çok dilli SentenceTransformer modeliyle vektöre dönüştürülür.
3. Metinler, embedding’ler ve dosya özeti SQLite içinde saklanır.
4. Dosya değişmediyse embedding’ler yeniden hesaplanmaz; değiştiyse eski indeks tamamen yenilenir.
5. Soru embedding’i ile cosine similarity hesaplanır, en iyi üç sonuç sıralanır ve varsayılan `0.38` eşiğinin altındakiler elenir. Türkçe embedding’in düşük puanladığı açık konu eşleşmelerinde (ör. “iade”) kontrollü sözcük desteği kullanılır; “fiyat” gibi metinde bulunmayan odaklar bu desteği alamaz.
6. Sonuç yoksa model çağrılmaz ve `Bu bilgi documents.txt dosyasında bulunamadı.` yanıtı verilir.
7. Sonuç varsa yalnızca bulunan metin parçaları Foundry Local sohbet modeline gönderilir.

## Teknolojiler ve mimari

- Python 3.12
- Streamlit
- Microsoft Foundry Local SDK 1.2.4
- `qwen2.5-1.5b` sohbet modeli
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` embedding modeli
- SQLite ve NumPy

Foundry Local kataloğunda `qwen3-embedding-0.6b` modeli de doğrulandı. Bu sunum projesinde Türkçe retrieval’ın kararlı ve daha düşük bellekle çalışması için mevcut çok dilli MiniLM modeli tercih edildi; böylece embedding ve sohbet için iki Foundry modeli aynı anda bellekte tutulmuyor.

```text
documents.txt
    ↓
ingest.py (okuma, chunking, embedding önbelleği)
    ↓
database.py (SQLite, cosine similarity, Top-K + threshold)
    ↓
app.py (Foundry Local model yaşam döngüsü ve sıkı prompt)
    ↓
app_ui.py (Streamlit)
```

Sohbet modeli tek noktadan [`settings.py`](settings.py) içindeki `CHAT_MODEL_NAME` değeriyle veya `LOCAL_RAG_CHAT_MODEL` ortam değişkeniyle değiştirilebilir.

## Kurulum (macOS / Apple Silicon)

Python 3.12 kurulu değilse Homebrew ile kurun:

```bash
brew install python@3.12
```

Proje klasöründe ortamı oluşturun ve bağımlılıkları kurun:

```bash
cd local-rag-project
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`foundry-local-sdk` paketi macOS/Apple Silicon için yerel Foundry çalışma zamanını da kurar. Uygulama ilk gerçek soruda `qwen2.5-1.5b` modelini indirip yükler. Model `data/model_cache/` altında önbelleğe alındıktan sonra yeniden indirilmez. İlk indirme için internet ve yeterli disk alanı gerekir; sonraki çıkarım yerel yapılır.

## Çalıştırma

```bash
source .venv/bin/activate
streamlit run app_ui.py
```

Tarayıcıdaki üç bölüm:

- **Soru-Cevap:** Soru sorun, yanıtı ve kullanılan parçaları görün.
- **Bilgi Metni:** `documents.txt` içeriğini düzenleyip **Kaydet ve indeksle** düğmesine basın.
- **Proje Hakkında:** Akış ve kullanılan modelleri görün.

Metin kaydedildiğinde indeks otomatik yenilenir ve eski sohbet geçmişi temizlenir. Uygulama `documents/` gibi başka klasörleri taramaz.

## Örnek sorular

- Nova Teknoloji ne zaman ve nerede kuruldu?
- Şirket hangi hizmetleri sunuyor?
- Çalışma saatleri nelerdir?
- Müşteri desteğine nasıl ulaşabilirim?
- İade süresi kaç gündür?
- Müşteri bilgileri üçüncü kişilerle paylaşılır mı?
- Şirketin sahibi kimdir?
- Ürünlerin fiyatı ne kadar?

Son iki sorunun bilgi metninde cevabı olmadığı için sabit “bulunamadı” mesajı dönmelidir.

## Test

```bash
python -m pytest -q
python -m py_compile app.py app_ui.py database.py ingest.py settings.py
```

## Gizlilik ve bilinen sınırlamalar

- `documents.txt`, SQLite veritabanı, embedding çıkarımı ve sohbet çıkarımı yerel cihazda çalışır.
- İlk SentenceTransformer ve Foundry model indirmeleri internet gerektirir.
- Eşik tabanlı semantic retrieval kusursuz değildir; çok belirsiz veya bilgi metnine sözcüksel olarak benzeyen ilgisiz sorular yanlış parça seçebilir.
- Küçük sohbet modeli, verilen parçayı özetlerken ifade hatası yapabilir. Kullanılan parça ve benzerlik puanı bu nedenle arayüzde gösterilir.
- Büyük dosyalar için tüm embedding’leri bellekte karşılaştıran bu sade SQLite yaklaşımı ölçeklenmez.
- Proje “sıfır halüsinasyon” garantisi vermez; bağlam dışı çağrıları azaltmak için retrieval eşiği ve sıkı prompt kullanır.

Başlıca referanslar: [Microsoft Foundry Local Python SDK](https://github.com/microsoft/Foundry-Local/tree/main/sdk/python)
