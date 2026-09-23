// Supabase bağlantı ayarları — üyelik + cihazlar arası senkron.
// İki değer de Supabase panelinde: Project Settings → API.
// "anon public" anahtarı tarayıcıda açıkça durması için tasarlanmıştır; veriyi
// Row Level Security korur (supabase/schema.sql). service_role anahtarını
// ASLA buraya yazma.
// Boş bırakılırsa site çalışmaya devam eder, "Giriş / Kayıt Ol" butonu
// üyeliğin henüz etkin olmadığını söyler.
window.BETAVUS_AUTH_CONFIG = {
  supabaseUrl: 'https://ttitipsmexqjsdcnpkwo.supabase.co',
  supabaseAnonKey: 'sb_publishable_M0aqu_AHhdHMuIcPYAS0_w_VJ_ftKGY'   // publishable (herkese açık) anahtar
};
