import streamlit as st
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
import pandas as pd
import aiohttp
import asyncio
import nest_asyncio
import re
from collections import Counter
from st_aggrid import AgGrid, GridOptionsBuilder
import io

from heredacode import (
    integrate_clustering_with_keywords,
    clean_text_for_clustering,
    is_unimportant_sentence,
    load_spelling_corrections,
    get_sentence_model,
    build_spelling_pattern,
    merge_similar_topics,
    find_optimal_clusters,
    find_question_variations,
    generate_representative
)

nest_asyncio.apply()

# LOAD CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# KONFIGURASI
api_id = int(st.secrets["API_ID"])
api_hash = st.secrets["API_HASH"]
session_name = "new_session"
wib = ZoneInfo("Asia/Jakarta")

# Load model
# sentence_model = get_sentence_model()

# Load spelling corrections
spelling = load_spelling_corrections('kata_baku.csv')

# Topik dan keyword
topik_keywords = {
    # Topik dengan logika "DAN" (semua kata harus ada)
    "Status Bast": [
        ["bast"],
        ["stuck", "bast"]
    ],
    "Verifikasi Toko": [
        ["verifikasi", "toko"],
        ["verivikasi", "toko"],
        ["cek", "id", "toko"],
        ["nib"]
    ],
    "Verifikasi Pembayaran": [
        ["verifikasi", "pembayaran"],
        ["verifikasi", "pesanan"],
        ["verivikasi", "pembayaran"],
        ["minta", "verifikasi"],
        ["konfirmasi"],
        ["notif", "error"],
        ["verifikasi"],
        ["verivikasi"]
    ],
    "Penerusan Dana": [
        ["penerusan", "dana"],
        ["dana", "diteruskan"],
        ["uang", "diteruskan"],
        ["penerusan"],
        ["diteruskan"],
        ["meneruskan"],
        ["dana", "teruskan"],
        ["uang", "teruskan"],
        ["penyaluran"],
        ["di teruskan"],
        ["salur"]
    ],
    "Dana Belum Masuk": [
        ["dana", "belum", "masuk"],
        ["uang", "belum", "masuk"],
        ["dana", "masuk", "belum"],
        ["uang", "masuk", "belum"],
        ["dana", "tidak", "masuk"],
        ["uang", "tidak", "masuk"],
        ["dana", "gagal", "masuk"],
        ["uang", "gagal", "masuk"],
        ["belum", "masuk", "rekening"],
        ["belum", "transfer", "masuk"],
        ["belum", "masuk"]
    ],
    "Jadwal Cair Dana": [
        ["bos", "cair"],
        ["bop", "cair"],
        ["jadwal", "cair"],
        ["kapan", "cair"],
        ["gelombang", "2"],
        ["tahap", "2"],
        ["pencairan"]
    ],
    "Kendala Akses" : [
        ["kendala", "akses"],
        ["gagal", "akses"],
        ["tidak", "bisa", "akses"],
        ["tidak", "bisa", "login"],
        ["tidak", "bisa", "masuk"],
        ["gagal", "login"],
        ["gagal", "masuk"],
        ["gagal", "akses"],
        ["reset", "akun"],
        ["reset", "password"],
        ["ganti", "password"],
        ["ganti", "akun"],
        ["ganti", "email"],
        ["ganti", "no hp"],
        ["ganti", "no telepon"],
        ["ganti", "telepon"],
        ["eror", "akses"],
        ["eror", "login"],
        ["eror"],
        ["web", "dibuka"],
        ["gk", "bisa", "masuk"],
        ["belum", "lancar"],
        ["bisa", "diakses"],
        ["gangguan"],
        ["gangguannya"],
        ["belum", "normal", "webnya"],
        ["trobel"],
        ["trobelnya"],
        ["ga", "bisa", "akses"],
        ["ga", "bisa", "log", "in"],
        ["ga", "bisa", "masuk"],
        ["ga", "bisa", "web"],
        ["g", "masuk2"],
        ["gk", "bisa2"],
        ["web", "troubel"],
        ["jaringan"],
        ["belum", "bisa", "masuk", "situs"],
        ["belum", "normal", "web"],
        ["vpn"],
        ["gabisa", "login"],
        ["gabisa", "akses"],
        ["g", "bisa", "akses"],
        ["g", "bisa", "login"],
        ["tidak", "bisa", "di", "buka"],
        ["bermasalah", "login"],
        ["login", "trouble"],
        ["sedang", "maintenance"],
        ["di block"],
        ["normal"],
        ["error"],
        ["trouble"],
        ["maintainance"]
    ],
    "Kendala Autentikasi": [
        ["kendala", "autentikasi"],
        ["gagal", "autentikasi"],
        ["tidak", "bisa", "autentikasi"],
        ["gagal", "otentikasi"],
        ["tidak", "bisa", "otentikasi"],
        ["authenticator", "reset"], 
        ["autentikasi", "salah"],
        ["autentikasi", "2", "langkah"],
        ["autentikasi", "dua", "langkah"],
        ["2", "langkah"],
        ["autentifikasi"],
        ["otentikasi"],
        ["otp", "gagal"],
        ["otp", "tidak", "bisa"],
        ["otp", "tidak", "muncul"],
        ["otp", "tidak", "tampil"],
        ["otp", "tidak", "ada"],
        ["reset", "barcode"],
        ["authenticator"],
        ["aktivasi"],
        ["otentikasi"]
    ],
    "Kendala Upload": [
        ["kendala", "upload"],
        ["gagal", "upload"],
        ["tidak", "bisa", "upload"],
        ["gagal", "unggah"],
        ["tidak", "bisa", "unggah"],
        ["produk", "tidak", "muncul"],
        ["produk", "tidak", "tampil"],
        ["produk", "tidak", "ada"],
        ["produk", "massal"],
        ["produk", "masal"],
        ["template", "upload"],
        ["template", "unggah"],
        ["unggah", "produk"],
        ["menambahkan", "produk"],
        ["menambah", "produk"],
        ["tambah", "produk"],
        ["tambah", "barang"],
        ["unggah", "foto"],
        ["unggah", "gambar"],
        ["unggah", "foto", "produk"],
        ["unggah", "gambar", "produk"],
        ["upload", "produk"],
        ["upload", "barang"]
    ],
    "Kendala Pengiriman": [
        ["tidak", "bisa", "pengiriman"],
        ["barang", "rusak"],
        ["barang", "hilang"],
        ["status", "pengiriman"]
    ],
    "Tanda Tangan Elektronik (TTE)": [
        ["tanda", "tangan", "elektronik"],
        ["ttd", "elektronik"],
        ["tte"],
        ["ttd"],
        ["tt elektronik"],
        ["e", "sign"],
        ["elektronik", "dokumen"],
        ["ttd", "elektronic"]
    ],
    "Ubah Data Toko": [
        ["ubah", "data", "toko"],
        ["edit", "data", "toko"],
        ["ubah", "nama", "toko"],
        ["edit", "nama", "toko"],
        ["ubah", "rekening"],
        ["edit", "rekening"],
        ["ubah", "status", "toko"],
        ["edit", "status", "toko"],
        ["ubah", "status", "umkm"],
        ["edit", "status", "umkm"],
        ["ubah", "status", "pkp"],
        ["ganti"]
    ],
    "Seputar Akun Pengguna": [
        ["ganti", "email"],
        ["ubah", "email"],
        ["ganti", "nama", "akun"],
        ["ubah", "nama", "akun"],
        ["ganti", "akun"],
        ["ubah", "akun"],
        ["gagal", "ganti", "akun"],
        ["gagal", "ubah", "akun"]
    ],
    "Pengajuan Modal": [
        ["pengajuan", "modal"],
        ["ajukan", "modal"],
        ["modal", "kerja"],
        ["dana", "talangan"],
        ["dibatalkan", "pengajuan"],
        ["tidak", "bisa", "ajukan"],
        ["modal", "talangan"],
        ["modal", "kerja"],
        ["dana", "talangan"],
        ["dana", "kerja"],
        ["modal", "bantuan"],
        ["modal", "usaha"],
        ["modal", "bantuan", "usaha"]
    ],
    "Pajak": [
        ["pajak", "ppn"],
        ["pajak", "invoice"],
        ["pajak", "npwp"],
        ["pajak", "penghasilan"],
        ["e-billing"],
        ["dipotong", "pajak"],
        ["pajak", "keluaran"],
        ["potongan", "pajak"],
        ["coretax"],
        ["pajak"],
        ["ppn"],
        ["npwp"],
        ["e-faktur"],
        ["efaktur"],
        ["e-billing"],
        ["dpp"],
        ["pph"]
    ],
    "Etika Penggunaan": [
        ["bendahara", "dapat", "untung"],
        ["bendahara", "dagang"],
        ["bendahara", "etik"],
        ["distributor", "dilarang"],
        ["etik", "distributor"],
        ["etik", "larangan"],
        ["etik", "juknis"],
        ["larangan"]
    ],
    "Waktu Proses": [
        ["kapan"],
        ["estimasi"],
        ["waktu", "proses"],
        ["waktu", "penyelesaian"],
        ["waktu", "selesai"],
        ["berapa","lm"],
        ["berapa", "lama"],
        ["berapa", "hari"],
        ["jam", "berapa"]
    ],
    "Pembatalan Pesanan": [
        ["batalkan", "pesanan"],
        ["pembatalan", "pesanan"],
        ["batalkan", "order"],
        ["pembatalan", "order"],
        ["batalin", "pesanan"],
        ["batalin", "order"],
        ["cancel"]
    ],
    
    # Topik dengan logika "ATAU" (salah satu kata cukup)
    "Pembayaran Dana": ["transfer", "dana masuk", "pengembalian", "bayar", "pembayaran", "dana", "dibayar", "notif pembayaran", "transaksi", "expired"],
    "Pengiriman Barang": ["pengiriman", "barang rusak", "kapan dikirim", "status pengiriman", "diproses"],
    "Penggunaan Siplah": ["pakai siplah", "siplah", "laporan siplah", "pembelanjaan", "tanggal pembelanjaan", "ubah tanggal", "dokumen", "bisa langsung dipakai", "terhubung arkas"],
    "Kurir Pengiriman": ["ubah kurir", "ubah jasa kirim", "jasa pengiriman", "jasa kurir", "kurir"],
    "Status": ["cek"],
    "Bantuan Umum": ["ijin tanya", "minta tolong", "tidak bisa", "cara", "masalah", "mau tanya", "input", "pkp", "pesanan gantung", "di luar dari arkas", "di bayar dari"],
    "lainnya": []
}

# TAMPILAN STREAMLIT 
st.set_page_config(page_title="Scraper & Analisis Telegram", layout="wide")
st.title("Analisis Topik Pertanyaan Grup Telegram")

# Input grup Telegram dan tanggal
group = st.text_input("Masukkan username atau ID grup Telegram:", "@contohgroup")
today = datetime.now(wib).date()
week_ago = today - timedelta(days=7)
col1, col2 = st.columns(2)
with col1:
    start_date_scrape = st.date_input("Scrape dari tanggal", week_ago)
with col2:
    end_date_scrape = st.date_input("Scrape sampai tanggal", today)

# Konversi date → datetime full (awal & akhir hari)
start_dt = datetime.combine(start_date_scrape, time.min).astimezone(wib)
end_dt   = datetime.combine(end_date_scrape, time.max).astimezone(wib)

# Fungsi pendukung
def is_question_like(text: str) -> bool:
    """Deteksi apakah teks valid pertanyaan dengan filter ketat."""
    if pd.isna(text) or not isinstance(text, str):
        return False
    
    text_lower = text.strip().lower()
    if not text_lower:
        return False

    # Filter noise awal 
    non_question_patterns = [
        r'^(min|admin|pak|bu|kk|kak|om|bro|sis)[\s\?]*$',  # cuma panggilan
        r'^[\?\.]+$',                                      # cuma tanda baca
        r'^(iya|ya+|oke+|ok|sip|noted+|oh+|lah+|loh+)$',   # respon singkat
        r'^(anggota baru|selamat berlibur|wah keren|mantap+|mantul+).*$',  # basa basi
    ]
    for pat in non_question_patterns:
        if re.match(pat, text_lower):
            return False

    # Cek tanda tanya (tetapi wajib ada >=3 kata bermakna)
    words = text_lower.split()
    if "?" in text_lower and len(words) >= 3:
        return True

    question_words = ["apa","apakah","siapa","kapan","mengapa","kenapa","bagaimana",
                      "gimana","dimana","berapa","kok","kenapakah","bagaimanakah"]
    if any(q in words for q in question_words):
        # wajib ada kata context (biar ga lolos yg no context)
        context_keywords = ["dana","uang","pembayaran","verifikasi","akun","login","akses","upload",
                            "barang","produk","toko","pengiriman","rekening","modal","npwp","pajak"]
        if any(kw in text_lower for kw in context_keywords):
            return True
        else:
            return False
        
    question_phrases = [
        "ada yang tahu", "mau tanya", "izin bertanya", "boleh tanya",
        "butuh bantuan", "ada solusi", "minta saran", "rekomendasi",
        "sudah diproses belum", "kok belum", "kapan cair", "gimana prosesnya",
        "cek status", "caranya gimana", "kenapa gagal"
    ]
    if any(phrase in text_lower for phrase in question_phrases):
        return True

    return False

async def scrape_messages(group, start_dt, end_dt, max_estimate=5000):
    all_messages = []
    sender_cache = {}

    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Menghubungkan ke Telegram...")

    try:
        async with TelegramClient(session_name, api_id, api_hash) as client:
            entity = await client.get_entity(group)
            offset_id = 0
            limit = 100
            total_fetched = 0
            stop_loop = False

            while True:
                history = await client(GetHistoryRequest(
                    peer=entity,
                    limit=limit,
                    offset_id=offset_id,
                    offset_date=None,   
                    add_offset=0,
                    max_id=0,
                    min_id=0,
                    hash=0
                ))

                messages = history.messages
                if not messages:
                    break

                for msg in messages:
                    if not getattr(msg, 'message', None) or not getattr(msg, 'date', None):
                        continue

                    msg_date_wib = msg.date.astimezone(wib)

                    # FILTER ketat
                    if msg_date_wib < start_dt:
                        stop_loop = True
                        break
                    if msg_date_wib > end_dt:
                        continue

                    sender_id = msg.sender_id
                    sender_name = sender_cache.get(sender_id)
                    if not sender_name:
                        try:
                            sender = await client.get_entity(sender_id)
                            sender_name = f"{sender.first_name or ''} {sender.last_name or ''}".strip() or sender.username or f"User ID: {sender_id}"
                        except Exception:
                            sender_name = f"User ID: {sender_id}"
                        sender_cache[sender_id] = sender_name

                    all_messages.append({
                        'id': msg.id,
                        'sender_id': sender_id,
                        'sender_name': sender_name,
                        'text': msg.message,
                        'date': msg_date_wib.strftime("%Y-%m-%d %H:%M:%S")
                    })

                total_fetched = len(all_messages)
                progress = min(1.0, total_fetched / max_estimate)
                progress_bar.progress(progress)
                progress_text.text(f"Mengambil pesan... {total_fetched}")

                if stop_loop:
                    break

                offset_id = messages[-1].id
                await asyncio.sleep(0)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat scraping: {e}")
        return None

    progress_bar.progress(1.0)
    progress_text.empty()
    st.success(f"Selesai mengambil {len(all_messages)} pesan!")
    return pd.DataFrame(all_messages)

def analyze_all_topics(df_questions):
    if df_questions.empty:
        st.warning("Tidak ada data pertanyaan yang bisa dianalisis.")
        return None, None

    num_messages = len(df_questions)
    if num_messages <= 50:
        num_auto_clusters = find_optimal_clusters(df_questions['processed_text'].tolist(), 2, 4)
    elif num_messages <= 100:
        num_auto_clusters = find_optimal_clusters(df_questions['processed_text'].tolist(), 3, 6)
    elif num_messages <= 200:
        num_auto_clusters = find_optimal_clusters(df_questions['processed_text'].tolist(), 4, 8)
    elif num_messages <= 300:
        num_auto_clusters = find_optimal_clusters(df_questions['processed_text'].tolist(), 5, 9)
    elif num_messages <= 500:
        num_auto_clusters = find_optimal_clusters(df_questions['processed_text'].tolist(), 7, 12)
    elif num_messages <= 1000:
        num_auto_clusters = find_optimal_clusters(df_questions['processed_text'].tolist(), 10, 18)
    else:
        num_auto_clusters = find_optimal_clusters(df_questions['processed_text'].tolist(), 15, 30)

    df_for_clustering = df_questions.copy()
    df_for_clustering["text"] = df_for_clustering["processed_text"]

    # Clustering topik utama 
    result = integrate_clustering_with_keywords(
        df_for_clustering,
        topik_keywords,
        spelling_corrections=spelling,
        num_auto_clusters=num_auto_clusters,
    )
    df_questions_with_topics = result[0] if isinstance(result, tuple) else result

    # Gabungkan topik mirip
    df_questions_with_topics = merge_similar_topics(
        df_questions_with_topics,
        sim_threshold=0.85,
        min_topic_size=3,
        use_embeddings=False
    )

    summary_clusters = {}
    for topik in df_questions_with_topics["final_topic"].unique():
        pertanyaan_topik = df_questions_with_topics.loc[
            df_questions_with_topics["final_topic"] == topik, "text"
        ].tolist()
        summary_clusters[topik] = pertanyaan_topik

    new_count = df_questions_with_topics.loc[
        df_questions_with_topics["final_topic"].str.lower().str.startswith("(new)")
    ]["final_topic"].nunique()

    st.subheader("Ringkasan Topik Teratas")
    st.markdown(f"**Jumlah topik baru terdeteksi: {new_count}**")

    topik_counter = Counter(df_questions_with_topics["final_topic"])
    summary_data = [
        {"Topik": topik, "Jumlah Pertanyaan": count}
        for topik, count in topik_counter.most_common()
    ]
    st.dataframe(pd.DataFrame(summary_data), width="stretch")

    st.subheader("Detail Pertanyaan per Topik")
    for topik, count in topik_counter.most_common():
        with st.expander(f"Topik: {topik} ({count} pertanyaan)"):
            questions_for_topic = df_questions_with_topics[
                df_questions_with_topics["final_topic"] == topik
            ]["text"].tolist()

            for q in questions_for_topic[:8]:
                st.markdown(f"- {q.strip()}")

    return df_questions_with_topics, summary_clusters

# Tombol eksekusi 
if st.button("Mulai Proses dan Analisis"):
    if not group or group == "@contohgroup":
        st.warning("⚠ Mohon isi nama grup Telegram yang valid terlebih dahulu.")
        st.stop()
    start_dt = datetime.combine(start_date_scrape, datetime.min.time()).replace(tzinfo=wib)
    end_dt = datetime.combine(end_date_scrape, datetime.max.time()).replace(tzinfo=wib)
    df_all = asyncio.run(scrape_messages(group, start_dt, end_dt))

    with st.spinner("Memuat model..."):
        sentence_model = get_sentence_model()
    st.success("Model berhasil dimuat!")

    if df_all is not None and not df_all.empty:
        df_all = df_all.sort_values('date').reset_index(drop=True)
        df_all['text'] = df_all['text'].str.lower()
        df_all['text'] = df_all['text'].str.replace(r'http\S+|www\.\S+', '', regex=True)
        df_all = df_all[df_all['text'].str.strip() != '']
        df_all.drop_duplicates(subset=['sender_id', 'text', 'date'], keep='first', inplace=True)
        df_all = df_all[~df_all['sender_name'].isin(['CS TokoLadang', 'Eko | TokLa', 'Vava'])]

        df_all['is_question'] = df_all['text'].apply(is_question_like)
        df_questions = df_all[df_all['is_question']].copy()

        # --- Load spelling corrections ---
        spelling = load_spelling_corrections("kata_baku.csv")   
        apply_spelling = build_spelling_pattern(spelling)      
        # --- Preprocessing dengan spelling ---
        df_questions['processed_text'] = df_questions['text'].apply(lambda x: clean_text_for_clustering(x, apply_spelling))
        df_questions = df_questions[~df_questions['processed_text'].apply(is_unimportant_sentence)]

        tab1, tab2, tab3 = st.tabs(["**Daftar Pertanyaan**", "**Analisis Topik**", "**Pertanyaan Representatif**"])

        with tab1:
            st.subheader(f"Ditemukan {len(df_questions)} Pesan Pertanyaan")

            if not df_questions.empty:
                df_show = df_questions[['date', 'sender_name', 'text']]

                gb = GridOptionsBuilder.from_dataframe(df_show)

                # Skala 1:1:3
                gb.configure_column("date", header_name="Tanggal", flex=1, minWidth=120, resizable=False, suppressMovable=True)
                gb.configure_column("sender_name", header_name="Pengirim", flex=1, wrapText=True, autoHeight=True, minWidth=150, resizable=False, suppressMovable=True)
                gb.configure_column("text", header_name="Pertanyaan", flex=3, wrapText=True, autoHeight=True, minWidth=300, resizable=False, suppressMovable=True)

                grid_options = gb.build()

                AgGrid(
                    df_show,
                    gridOptions=grid_options,
                    height=500,
                    fit_columns_on_grid_load=True,   
                    enable_enterprise_modules=False,
                    allow_unsafe_jscode=True,
                    theme="streamlit",
                    reload_data=True,
                    custom_css={
                        ".ag-root-wrapper": {"width": "100% !important"},
                        ".ag-theme-streamlit": {
                            "width": "100% !important",
                            "overflow": "hidden !important",  
                        },
                    },
                    update_mode="MODEL_CHANGED",
                    suppressHorizontalScroll=True,   
                )
            else:
                st.info("Tidak ada pesan yang terdeteksi sebagai pertanyaan pada periode ini.")

        with tab2:
            df_questions_with_topics, summary_clusters = analyze_all_topics(df_questions)

        with tab3:
            st.subheader("Pertanyaan Representatif per Variasi Topik")
        
            if df_questions_with_topics is None or df_questions_with_topics.empty:
                st.warning("Belum ada hasil analisis topik untuk dibuat representatifnya.")
                st.stop()
        
            st.markdown("Sistem akan memecah setiap topik menjadi **beberapa variasi pertanyaan**, lalu membuat **kalimat tanya formal** untuk setiap variasi tersebut.")
        
            all_topics = df_questions_with_topics["final_topic"].unique().tolist()
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            async def process_all_topics_async(all_topics, df_questions_with_topics):
                semaphore = asyncio.Semaphore(5) 
                async def bounded_generate_representative(session, variation_questions):
                    async with semaphore: # Tunggu giliran, lalu kirim permintaan
                        return await generate_representative_async(session, variation_questions)
            
                async with aiohttp.ClientSession() as session:
                    tasks = []
                    task_metadata = []
                    for i, topik in enumerate(all_topics):
                        progress_text.text(f"Menyiapkan task untuk topik {i+1}/{len(all_topics)}: {topik}")
                        
                        questions_in_topic = df_questions_with_topics[
                            df_questions_with_topics["final_topic"] == topik
                        ]["text"].tolist()
            
                        if not questions_in_topic:
                            continue
            
                        variations = find_question_variations(questions_in_topic, min_variation_size=3)
                        for j, variation_questions in enumerate(variations):
                            task = asyncio.create_task(bounded_generate_representative(session, variation_questions))
                            tasks.append(task)
                            task_metadata.append({
                                "Topik Utama": topik,
                                "Jumlah Pertanyaan di Variasi": len(variation_questions),
                                "Pertanyaan Asli": variation_questions
                            })
            
                    # Jalankan semua task secara paralel
                    progress_text.text("Menjalankan semua permintaan ke AI secara paralel...")
                    results = await asyncio.gather(*tasks, return_exceptions=True)
            
                    # Gabungkan hasil dengan metadata
                    final_results = []
                    for i, res in enumerate(results):
                        if isinstance(res, Exception):
                            print(f"Error pada topik {task_metadata[i]['Topik Utama']}: {res}")
                            representative_sentence = smart_embedding_fallback(task_metadata[i]['Pertanyaan Asli'])
                        else:
                            representative_sentence = res
                        
                        final_results.append({
                            **task_metadata[i],
                            "Kalimat Representatif (AI)": representative_sentence
                        })
                    
                    return final_results

            final_results = asyncio.run(process_all_topics_async(all_topics, df_questions_with_topics))
            
            progress_bar.empty()
            progress_text.empty()
        
            if not final_results:
                st.info("Tidak ada variasi pertanyaan yang cukup signifikan untuk dianalisis.")
                st.stop()
            
            # Kelompokkan berdasarkan topik utama
            df_results = pd.DataFrame(final_results)
            grouped = df_results.groupby("Topik Utama")
        
            for topik_name, group_df in grouped:
                with st.expander(f"{topik_name}", expanded=False):
                    for _, row in group_df.iterrows():
                        st.markdown(
                            f"""
                            <div style="padding: 10px; border-left: 4px solid #E0935A; background-color: #f9f9f9; margin-bottom: 10px; border-radius: 5px;">
                                <strong>Kalimat Representatif:</strong> {row['Kalimat Representatif (AI)']} 
                                <span style="color: grey; font-size: 0.9em;">({row['Jumlah Pertanyaan di Variasi']} pertanyaan)</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        with st.expander("Lihat pertanyaan asli yang menjadi dasar kalimat ini"):
                            for q in row['Pertanyaan Asli']:
                                st.markdown(f"- {q.strip()}")
                    
            # Tombol Download
            output = io.BytesIO()
            df_download = df_results.drop(columns=['Pertanyaan Asli'])
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_download.to_excel(writer, sheet_name='Representatif', index=False)
            output.seek(0)
        
            st.download_button(
                label="📥 Download Hasil Representatif (Excel)",
                data=output,
                file_name=f"hasil_representatif_variasi_{datetime.now(wib).strftime('%Y-%m-%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )



