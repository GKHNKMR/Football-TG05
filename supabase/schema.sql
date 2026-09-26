-- BETAVUS üyelik + senkron şeması (Supabase SQL Editor'de bir kez çalıştır).
-- Tekrar çalıştırmak güvenlidir.

create extension if not exists pgcrypto with schema extensions;

-- 1) Profil: kayıtta sorulan basit bilgiler ------------------------------------
create table if not exists public.profiles (
  id          uuid primary key references auth.users(id) on delete cascade,
  username    text check (username ~ '^[A-Za-z0-9_.]{3,20}$'),
  age         int  check (age between 18 and 100),
  gender      text check (gender in ('erkek','kadin','belirtmek_istemiyorum')),
  country     text check (char_length(country) between 2 and 10),
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);
create unique index if not exists profiles_username_lower on public.profiles (lower(username));

alter table public.profiles enable row level security;
drop policy if exists "profiles_select_own" on public.profiles;
drop policy if exists "profiles_insert_own" on public.profiles;
drop policy if exists "profiles_update_own" on public.profiles;
create policy "profiles_select_own" on public.profiles for select using (auth.uid() = id);
create policy "profiles_insert_own" on public.profiles for insert with check (auth.uid() = id);
create policy "profiles_update_own" on public.profiles for update using (auth.uid() = id) with check (auth.uid() = id);

-- 2) Kullanıcı verisi: Sanal Kasa + tercihler, {anahtar: {v, t}} -----------------
create table if not exists public.user_state (
  user_id     uuid primary key references auth.users(id) on delete cascade,
  data        jsonb not null default '{}'::jsonb,
  updated_at  timestamptz not null default now()
);
alter table public.user_state enable row level security;
drop policy if exists "state_select_own" on public.user_state;
drop policy if exists "state_insert_own" on public.user_state;
drop policy if exists "state_update_own" on public.user_state;
create policy "state_select_own" on public.user_state for select using (auth.uid() = user_id);
create policy "state_insert_own" on public.user_state for insert with check (auth.uid() = user_id);
create policy "state_update_own" on public.user_state for update using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- 3) Yeni kullanıcıda profil satırını otomatik aç (e-posta kaydında bilgiler
--    signUp metadata'sından gelir; Google girişinde boş açılır, site tamamlatır)
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
declare m jsonb := coalesce(new.raw_user_meta_data, '{}'::jsonb);
begin
  begin
    insert into public.profiles (id, username, age, gender, country)
    values (new.id, nullif(m->>'username',''), nullif(m->>'age','')::int,
            nullif(m->>'gender',''), nullif(m->>'country',''))
    on conflict (id) do nothing;
  exception when others then
    -- kullanıcı adı çakışması vb.: boş profil aç, site tamamlatır
    insert into public.profiles (id) values (new.id) on conflict (id) do nothing;
  end;
  return new;
end $$;
drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created after insert on auth.users
  for each row execute function public.handle_new_user();

-- 4) Kullanıcı adı müsait mi?
create or replace function public.username_available(p_username text)
returns boolean language sql stable security definer set search_path = public as $$
  select not exists (select 1 from public.profiles where lower(username) = lower(p_username));
$$;

-- 5) Kullanıcı adıyla giriş: şifre doğruysa e-postayı döndürür (yanlışsa null),
--    böylece kullanıcı adından e-posta öğrenilemez. 15 dk'da 10 hatalı denemede kilitlenir.
create table if not exists public.login_attempts (
  username text not null,
  at       timestamptz not null default now()
);
create index if not exists login_attempts_idx on public.login_attempts (lower(username), at);
alter table public.login_attempts enable row level security;   -- politika yok: yalnızca aşağıdaki fonksiyon erişir

create or replace function public.email_for_login(p_username text, p_password text)
returns text language plpgsql security definer set search_path = public, extensions, auth as $$
declare v_email text;
begin
  if (select count(*) from public.login_attempts
      where lower(username) = lower(p_username) and at > now() - interval '15 minutes') >= 10 then
    raise exception 'too_many_attempts';
  end if;
  select u.email into v_email
    from auth.users u join public.profiles p on p.id = u.id
   where lower(p.username) = lower(p_username)
     and coalesce(u.encrypted_password, '') <> ''
     and u.encrypted_password = extensions.crypt(p_password, u.encrypted_password);
  if v_email is null then
    insert into public.login_attempts (username) values (lower(p_username));
    delete from public.login_attempts where at < now() - interval '1 day';
  end if;
  return v_email;
end $$;

-- 6) "Hesabımı sil": giriş yapmış kullanıcı kendi hesabını ve ona bağlı tüm veriyi siler.
--    auth.users satırı silinince profiles / user_state (on delete cascade), oturumlar ve
--    Google kimliği de gider. Başkasının hesabına dokunamaz: yalnızca auth.uid() silinir.
create or replace function public.delete_my_account()
returns void language plpgsql security definer set search_path = public, auth as $$
declare v_uid uuid := auth.uid();
begin
  if v_uid is null then raise exception 'not_authenticated'; end if;
  delete from public.login_attempts
   where lower(username) = (select lower(username) from public.profiles where id = v_uid);
  delete from public.user_state where user_id = v_uid;
  delete from public.profiles  where id = v_uid;
  delete from auth.users       where id = v_uid;
end $$;

revoke all on function public.delete_my_account() from public, anon;
grant execute on function public.delete_my_account() to authenticated;

revoke all on function public.email_for_login(text, text) from public;
revoke all on function public.username_available(text) from public;
grant execute on function public.email_for_login(text, text) to anon, authenticated;
grant execute on function public.username_available(text) to anon, authenticated;
