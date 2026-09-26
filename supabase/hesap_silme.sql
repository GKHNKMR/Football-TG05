-- "Hesabımı sil" (iş listesi #12) — mevcut kuruluma bir kez eklenir.
-- Supabase → SQL Editor → bu dosyanın tamamını yapıştır → Run. Tekrar çalıştırmak güvenlidir.
-- (Aynı blok schema.sql'in 6. bölümünde de var; sıfırdan kurulumda ayrıca çalıştırmaya gerek yok.)

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
