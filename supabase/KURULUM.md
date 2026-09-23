# BETAVUS üyelik kurulumu (Supabase + Google)

Kod hazır. Üyeliğin çalışması için bir kez aşağıdaki adımlar gerekiyor (~15 dk).
Bunlar yapılana kadar site normal çalışır; "Giriş / Kayıt Ol" butonu üyeliğin henüz etkin
olmadığını söyler.

## 1. Supabase projesi
1. https://supabase.com → ücretsiz hesap → **New project** (bölge: Frankfurt / `eu-central-1`).
2. **SQL Editor** → `supabase/schema.sql` dosyasının tamamını yapıştır → **Run**.
3. **Project Settings → API** sayfasından iki değeri kopyala:
   - `Project URL`
   - `anon public` anahtarı (**service_role anahtarını değil**)
4. `js/auth_config.js` içine yaz:
   ```js
   supabaseUrl: 'https://xxxx.supabase.co',
   supabaseAnonKey: 'eyJ...'
   ```
   `anon` anahtarı herkese açık olacak şekilde tasarlanmıştır; veriyi tablo kuralları (RLS) korur.

## 2. Adresler
**Authentication → URL Configuration**
- Site URL: `https://betavus.vercel.app`
- Redirect URLs: `https://betavus.vercel.app/**` (yerel test için ayrıca `http://localhost:8000/**`)

## 3. E-posta + şifre
**Authentication → Providers → Email**: açık. "Confirm email" açık kalsın (kayıtta doğrulama
bağlantısı gider). Supabase'in ücretsiz e-posta gönderimi saatte birkaç e-postayla sınırlıdır;
kullanıcı sayısı artınca **Project Settings → Auth → SMTP** ile kendi SMTP'ni bağla.

## 4. Google ile giriş
1. https://console.cloud.google.com → proje seç/oluştur → **APIs & Services → OAuth consent screen**
   → External → uygulama adı "BETAVUS", destek e-postası → kaydet → **Publish app**.
2. **Credentials → Create credentials → OAuth client ID** → *Web application*.
   - Authorized JavaScript origins: `https://betavus.vercel.app`
   - Authorized redirect URIs: `https://<proje-ref>.supabase.co/auth/v1/callback`
     (tam adres Supabase'de Google sağlayıcı sayfasında yazıyor)
3. Çıkan **Client ID** ve **Client secret**'ı Supabase → **Authentication → Providers → Google**
   içine yapıştır → Enable → Save.

## Nasıl çalışıyor
- Kayıt: kullanıcı adı, yaş (18+), cinsiyet, ülke, e-posta, şifre. Google ile gelenlere ilk
  girişte aynı dört basit soru sorulur.
- Giriş: kullanıcı adı **veya** e-posta + şifre, ya da Google.
- Senkron: tarayıcıdaki tüm `betavus.*` verileri (Sanal Kasa `betavus.paper_v1`, tercihler)
  `user_state` tablosunda tutulur. Her değişiklik ~1,5 sn içinde buluta yazılır; sekmeye
  dönüldüğünde ve dakikada bir diğer cihazların değişiklikleri çekilir. Yeni olan kazanır.
- Veri kaybına karşı: bir cihaz hesaba ilk bağlandığında o cihazdaki kasalar/kuponlar
  buluttakilerle **birleştirilir**, üzerine yazılmaz. Üzerine yazılan her yerel değerin son 3
  kopyası tarayıcıda `betavus.__backup.*` altında saklanır.
- Erişim kodu kapısı olduğu gibi duruyor; üyelik onun arkasında çalışır.
