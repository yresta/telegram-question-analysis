import re
import os
from typing import List, Dict, Any, Tuple
from functools import lru_cache
import requests

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from typing import Callable, Optional, List

# Setup NLTK
nltk.download('punkt', quiet=True)

# Configuration / Constants
DEFAULT_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
MIN_CLUSTER_SIZE = 8
MAX_RECURSIVE_DEPTH = 3
EMBEDDING_BATCH_SIZE = 128
model = SentenceTransformer(DEFAULT_MODEL_NAME)

IND_STOPWORDS = set("""
yang dan di ke dari untuk dengan pada oleh dalam atas sebagai adalah ada itu ini atau tidak sudah belum bisa akan harus sangat juga karena jadi kalau namun tapi serta agar supaya sehingga maka lalu kemudian setelah sebelum hingga sampai pun saya kak bapak ibu pak
""".split())

QUESTION_WORDS = set("""apa siapa kapan mengapa kenapa bagaimana gimana dimana apakah mana dimana saja kenapa ya tidak""".split())

FOCUS_KEYWORDS = {
    "dana","pencairan","cair","rekening","saldo","uang",
    "pembayaran","bayar","verifikasi","otp","autentikasi",
    "login","akses","akun","aplikasi",
    "produk","barang","pesanan","kurir","pengiriman","retur",
    "ppn","pajak","modal","talangan","data","toko","upload","unggah"
}

# Sentence model (cached globally)
@lru_cache(maxsize=1)
def get_sentence_model(model_name: str = DEFAULT_MODEL_NAME):
    return SentenceTransformer(model_name)

def get_sentence_embeddings(texts: List[str], model=None, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384))
    if model is None:
        model = get_sentence_model(model_name)
    return np.asarray(model.encode(
        texts,
        show_progress_bar=False,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True
    ))

# Spelling correction
def load_spelling_corrections(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        if 'tidak_baku' in df.columns and 'kata_baku' in df.columns:
            return dict(zip(df['tidak_baku'].astype(str), df['kata_baku'].astype(str)))
    except Exception:
        pass
    return {}

def build_spelling_pattern(corrections: Dict[str, str]):
    if not corrections:
        return None
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in corrections.keys()) + r')\b')
    return lambda text: pattern.sub(lambda m: corrections[m.group(0)], text)

spelling = load_spelling_corrections("kata_baku.csv")
apply_spelling = build_spelling_pattern(spelling)

# Cleaning & filtering
def is_unimportant_sentence(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return True
    txt = text.strip().lower()

    # kalau cuma tanda baca doang
    if re.fullmatch(r"[\W_]+", txt):
        return True

    # kalau mayoritas isi tanda baca (>= 70%)
    punct_ratio = sum(1 for c in txt if not c.isalnum()) / max(len(txt), 1)
    if punct_ratio > 0.7:
        return True
    
    unimportant_phrases = {
        "siap","noted","oke","ok","baik","sip","thanks","makasih","terima kasih",
        "iya","ya","oh","ohh","mantap","mantul","keren","wah","hebat",
        "anggota baru","selamat berlibur"
    }
    if txt in unimportant_phrases:
        return True
    if len(txt.split()) <= 2 and not any(q in txt for q in QUESTION_WORDS):
        return True
    return False

def clean_text_for_clustering(text: Any, spelling_fn: Optional[Callable[[str], str]] = None) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    if spelling_fn:  
        text = spelling_fn(text)
    text = re.sub(r'[^0-9a-z\s\?%.,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clustering
def cluster_texts_embeddings(X: np.ndarray, num_clusters: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] == 0:
        return np.array([]), np.array([])
    k = min(max(1, int(num_clusters)), X.shape[0])
    if k == 1:
        labels = np.zeros(X.shape[0], dtype=int)
        centers = X.mean(axis=0, keepdims=True)
        return labels, centers
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)

    return labels, kmeans.cluster_centers_

def find_optimal_clusters(texts: List[str], min_k: int = 2, max_k: int = 10) -> int:
    if not texts or len(texts) < 2:
        return 1
    X = get_sentence_embeddings(texts)
    best_k, best_score = 1, -1
    max_k = min(max_k, len(texts))
    for k in range(min_k, max_k+1):
        try:
            labels = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=5).fit_predict(X)
            if len(set(labels)) == 1:
                continue
            score = silhouette_score(X, labels)
        except Exception:
            score = -1
        if score > best_score:
            best_k, best_score = k, score
    return best_k

def recursive_clustering(texts: List[str], embeddings=None, 
                         min_cluster_size: int = MIN_CLUSTER_SIZE,
                         max_depth: int = MAX_RECURSIVE_DEPTH,
                         depth: int = 0) -> List[List[str]]:
    if embeddings is None:
        embeddings = get_sentence_embeddings(texts)
    if depth >= max_depth or len(texts) <= min_cluster_size:
        return [texts]
    num_clusters = find_optimal_clusters(texts, min_k=2, max_k=min(10, len(texts)//min_cluster_size+1))
    if num_clusters <= 1:
        return [texts]

    labels, _ = cluster_texts_embeddings(embeddings, num_clusters=num_clusters)
    clusters = []
    for cid in set(labels):
        members = [texts[i] for i, lbl in enumerate(labels) if lbl == cid]
        sub_embeds = embeddings[labels == cid]
        if len(members) <= min_cluster_size or depth+1 >= max_depth:
            clusters.append(members)
        else:
            clusters.extend(recursive_clustering(members, sub_embeds, min_cluster_size, max_depth, depth+1))
    return clusters

# Keyword extraction
def extract_representative_keywords(texts: List[str], top_n: int = 2, max_features: int = 1000) -> List[str]:
    if not texts:
        return ["Topik"]

    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return ["Topik"]

    BAD_TOPIC_WORDS = {"sekolah", "bingung", "tolong", "mohon", "pak", "kak", "admin"}

    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1,2),  
            stop_words=IND_STOPWORDS
        )
        X = vectorizer.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        features = vectorizer.get_feature_names_out()
        sorted_idx = np.argsort(scores)[::-1]

        keywords: List[str] = []
        for i in sorted_idx:
            word = split_stuck_words(features[i])
            tokens = word.split()

            # skip kalau meaningless atau terlalu generik
            if not all(is_meaningful_word(tok) for tok in tokens):
                continue
            if any(tok in BAD_TOPIC_WORDS for tok in tokens):
                continue

            keywords.append(word)
            if len(keywords) >= top_n:
                break

        return keywords if keywords else ["Topik"]

    except Exception:
        tokens = ' '.join(texts).split()
        freqs = pd.Series(tokens).value_counts()
        return [
            w for w in freqs.index if is_meaningful_word(w) and w not in BAD_TOPIC_WORDS
        ][:top_n] or ["Topik"]

# Keyword extraction
def split_stuck_words(word: str) -> str:
    patterns = [
        (r'(verifikasi)(pembayaran)', r'\1 \2'),
        (r'(pembayaran)(dana)', r'\1 \2'),
        (r'(dana)(masuk)', r'\1 \2'),
        (r'(modal)(talangan)', r'\1 \2')
    ]
    for pat, repl in patterns:
        word = re.sub(pat, repl, word)
    return word

CUSTOM_IGNORE = {"kak","min","ya","kah","buk","pak", "min","om","yaa","apaaaaa","omom","loh","lah","deh"}

def is_meaningful_word(word: str) -> bool:
    word = word.strip()
    if len(word) <= 3 or word.isdigit():
        return False
    if word in IND_STOPWORDS or word in QUESTION_WORDS or word in CUSTOM_IGNORE:
        return False
    return True

spelling = load_spelling_corrections("kata_baku.csv")
apply_spelling = build_spelling_pattern(spelling)

# Normalisasi nama topik otomatis
CUSTOM_IGNORE = {
    "min","admin","kok","kah","map","kira","pernah","ya","izin",
    "sih","loh","lah","deh","dong","nih","aja","saja","selamat",
    "new"
}

TOPIC_IGNORE = {
    "seperti","lagi","adakah","kak","dong","nih","aja","saja",
    "min","admin","map","pernah","ya","izin","sih","loh","lah","deh",
    "bagaimana","gimana","kok","kah","dong","nih","aja","saja","masih"
}

GENERIC_IGNORE = {
    "apa","ini","itu","lagi","sudah","bisa","tidak","belum","akan",
    "ada","dapat","dengan","dari","untuk","pada","dilaporkan","kendala"
}

def auto_topic_name(texts: List[str], top_n: int = 3) -> str:
    if not texts:
        return "Lainnya"

    kws = extract_representative_keywords(texts, top_n=top_n)
    rep_sent = get_cluster_representative_sentence(texts)
    clean = re.sub(r"[^0-9a-zA-Z\s]", " ", " ".join(kws))
    tokens = [t for t in clean.lower().split() if len(t) > 2]

    QUESTION_BAD = {
        "apa","apakah","bagaimana","gimana","kenapa","mengapa","kok","ya","iya",
        "berapakah","berapa","dimana","mana","kah","kan","kira","seperti"
    }
    GENERIC_BAD = {
        "tolong","info","informasi","mohon","baru","harap","sudah","belum","ok",
        "oke","baik","nah","yah","min","admin","kak","pak","bu","bro","gan","dong",
        "lah","nih","aja","saja","teman","kita","halaman","awal","siang","pembeli",
        "new","solusinya","gunanya","maksudnya","knapa","terima", "toko"
    }

    tokens = [t for t in tokens if len(t) > 2
                                and re.search(r'[a-z]', t)
                                and t not in IND_STOPWORDS
                                and t not in CUSTOM_IGNORE
                                and t not in TOPIC_IGNORE
                                and t not in GENERIC_IGNORE
                                and t not in QUESTION_BAD
                                and t not in GENERIC_BAD]

    focus_hits = [t for t in tokens if t in FOCUS_KEYWORDS]
    if focus_hits:
        extra = [t for t in tokens if t not in focus_hits][:1]
        name = " ".join([w.title() for w in focus_hits[:2] + extra])
    elif tokens:
        name = " ".join(tokens[:3]).title()
    elif rep_sent:
        rep_tokens = [w.title() for w in rep_sent.split() if len(w) > 3][:3]
        name = " ".join(rep_tokens) if rep_tokens else "Lainnya"
    else:
        name = "Lainnya"

    BAD_TOKENS_EXTRA = {"Kak","Pak","Min","Admin","Gan","Bro",
                        "Solusinya","Gunanya","Maksudnya","kah",
                        "sama","tanggal","minggu","hari","lain",
                        "maaf","harga","coba","toko", "lainnya",
                        "buat","hanya","bulan","yang","kenapa",
                        "kata","marah","mesti","kayak","begini",
                        "kakak","terus"}
    
    for bad in BAD_TOKENS_EXTRA:
        name = re.sub(rf"\b{bad}\b", "", name, flags=re.I)

    name = re.sub(r"\?", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    if not name or name.lower() in {"", ",", "new"}:
        return "Lainnya"

    if name.lower().startswith("(new)"):
        return "(new) " + name[5:].strip()

    if name.lower().startswith("new "):
        return "(new) " + name[4:].strip()

    return "(new) " + name

def get_cluster_representative_sentence(texts: List[str]) -> str:
    if not texts:
        return ""
    embeds = get_sentence_embeddings(texts)
    centroid = embeds.mean(axis=0, keepdims=True)
    sims = cosine_similarity(embeds, centroid).ravel()
    best_idx = np.argmax(sims)
    return texts[best_idx]

def normalize_topic_name(topic: str | list[str], cluster_texts: list[str] = None) -> str:
    if isinstance(topic, list): 
        return auto_topic_name(topic)
    elif cluster_texts:  
        return auto_topic_name(cluster_texts)
    else:  
        return auto_topic_name([topic])

def is_bad_cluster(texts: List[str]) -> bool:
    if not texts or len(texts) <= 1:
        return True
    kws = extract_representative_keywords(texts, top_n=2)
    if kws == ["Topik"]:
        return True
    embeds = get_sentence_embeddings(texts)
    if embeds.shape[0] > 1:
        sims = np.dot(embeds, embeds.T)
        avg_sim = (np.sum(sims) - np.trace(sims)) / (embeds.shape[0]**2 - embeds.shape[0])
        if avg_sim < 0.25:  
            return True
    return False

def merge_bad_cluster(df, similarity_threshold=0.55):
    clusters = df['cluster'].unique()
    for c in clusters:
        texts = df[df['cluster'] == c]['text'].tolist()
        if is_bad_cluster(texts):
            embeds = get_sentence_embeddings(texts)
            centroid = np.mean(embeds, axis=0, keepdims=True)

            best_match, best_sim = None, -1
            for other in clusters:
                if other == c: 
                    continue
                other_texts = df[df['cluster'] == other]['text'].tolist()
                other_embeds = get_sentence_embeddings(other_texts)
                sim = np.dot(centroid, np.mean(other_embeds, axis=0)) / (
                    np.linalg.norm(centroid) * np.linalg.norm(np.mean(other_embeds, axis=0))
                )
                if sim > best_sim:
                    best_match, best_sim = other, sim

            if best_match is not None and best_sim > similarity_threshold:
                df.loc[df['cluster'] == c, 'cluster'] = best_match
            else:
                df.loc[df['cluster'] == c, 'cluster'] = -1  
    return df

GENERIC_TOPICS = {
    "knapa","kenapa","toko","bank","status","bertanya","siang",
    "kira","pembeli","lainnya"
}

# Integrasi keyword + clustering 
def integrate_clustering_with_keywords(df: pd.DataFrame,
                                       topik_keywords: Dict[str, Any],
                                       spelling_corrections: Dict[str, str] = None,
                                       min_cluster_size: int = MIN_CLUSTER_SIZE,
                                       max_recursive_depth: int = MAX_RECURSIVE_DEPTH,
                                       num_auto_clusters: int = 15
                                       ) -> pd.DataFrame:
    df = df.copy()
    df['original_index'] = df.index
    df['processed_text'] = df['text'].apply(lambda t: clean_text_for_clustering(t, apply_spelling))
    df['is_unimportant'] = df['processed_text'].apply(is_unimportant_sentence)
    df = df[~df['is_unimportant']].reset_index(drop=True)

    keyword_categorized, remaining = [], []
    for _, row in df.iterrows():
        idx, txt = int(row['original_index']), row['processed_text']
        matched = []
        for topik, patterns in topik_keywords.items():
            if not patterns:
                continue
            if isinstance(patterns, list) and patterns and isinstance(patterns[0], list):
                if any(all(pat in txt for pat in conj) for conj in patterns):
                    matched.append(topik)
            else:
                if any(pat in txt for pat in (patterns if isinstance(patterns, list) else [patterns])):
                    matched.append(topik)
        if matched:
            spesifik = [t for t in matched if t != 'bantuan_umum']
            keyword_categorized.append((idx, spesifik[0] if spesifik else matched[0]))
        else:
            remaining.append((idx, txt))

    auto_categorized = []
    if remaining:
        rem_texts = [t for _, t in remaining]
        rem_idxs = [i for i, _ in remaining]

        grouped_clusters = recursive_clustering(
            rem_texts,
            min_cluster_size=min_cluster_size,
            max_depth=max_recursive_depth
        )

        for group in grouped_clusters:
            group = [t for t in group if isinstance(t, str) and t.strip()]
            if not group:
                continue

            indices_in_remaining = [i for i, txt in enumerate(rem_texts) if txt in group and txt is not None]
            for pos in indices_in_remaining:
                rem_texts[pos] = None
            orig_indices = [rem_idxs[pos] for pos in indices_in_remaining]

            if is_bad_cluster(group):
                for i, txt in zip(orig_indices, group):
                    matched_topic = None
                    for topik, patterns in topik_keywords.items():
                        if not patterns:
                            continue

                        if isinstance(patterns, list) and patterns and isinstance(patterns[0], list):
                            if any(all(subpat in txt for subpat in conj) for conj in patterns):
                                matched_topic = topik
                                break
                        else:
                            if any(
                                isinstance(pat, str) and pat in txt
                                for pat in (patterns if isinstance(patterns, list) else [patterns])
                            ):
                                matched_topic = topik
                                break

                    auto_categorized.append((i, matched_topic if matched_topic else "Lainnya"))
            else:
                if len(group) <= min_cluster_size:
                    topic_name = auto_topic_name(group, top_n=2)
                    topic_name = normalize_topic_name(topic_name)  
                    auto_categorized.extend((i, topic_name) for i in orig_indices)
                else:
                    X = get_sentence_embeddings(group)
                    k = max(1, len(group) // min_cluster_size)
                    labels, _ = cluster_texts_embeddings(X, num_clusters=k)
                    topic_names = {lbl: auto_topic_name([group[i] for i, l in enumerate(labels) if l == lbl])for lbl in set(labels.tolist())}
                    for j, lbl in enumerate(labels.tolist()):
                        if j < len(orig_indices):
                            normalized_name = normalize_topic_name(topic_names[lbl])
                            auto_categorized.append((orig_indices[j], normalized_name))

    mapping = {}
    for idx, topic in keyword_categorized + auto_categorized:
        if idx not in mapping:
            mapping[idx] = topic

    df['final_topic'] = df['original_index'].apply(lambda i: mapping.get(i, 'Lainnya'))

    new_topics = {t for t in df['final_topic'].unique() if str(t).lower().startswith("(new)")}
    num_new_topics = len(new_topics)

    return df[['original_index', 'text', 'processed_text', 'final_topic']], new_topics, num_new_topics

def merge_similar_topics(df: pd.DataFrame, sim_threshold: float = 0.75, min_topic_size: int = 3, 
                         use_embeddings: bool = False) -> pd.DataFrame:
    if isinstance(df, tuple):
        df = df[0]
    if "final_topic" not in df.columns:
        print("⚠️ Tidak ada kolom 'final_topic'. Skip merge_similar_topics.")
        return df

    topics = [str(t).strip() for t in df["final_topic"].unique() if str(t).strip()]
    if len(topics) <= 1:
        return df

    if use_embeddings:
        embeds = get_sentence_embeddings(topics)
        sim_matrix = cosine_similarity(embeds)
    else:
        vectorizer = TfidfVectorizer().fit(topics)
        X = vectorizer.transform(topics)
        sim_matrix = cosine_similarity(X)

    merged_map = {}
    used = set()
    for i, t1 in enumerate(topics):
        if t1 in used:
            continue
        merged_map[t1] = t1
        for j, t2 in enumerate(topics):
            if i != j and sim_matrix[i, j] >= sim_threshold:
                merged_map[t2] = t1
                used.add(t2)

    df["final_topic"] = df["final_topic"].map(merged_map).fillna(df["final_topic"])

    counts = df["final_topic"].value_counts()
    small_topics = counts[counts < min_topic_size].index.tolist()
    if small_topics:
        df.loc[df["final_topic"].isin(small_topics), "final_topic"] = "Lainnya"

    def clean_topic_name(x):
        x = re.sub(r"\s+", " ", str(x)).strip()
        if not x.lower().startswith("(new)"):
            x = x.title()
        return x

    df["final_topic"] = df["final_topic"].apply(clean_topic_name)
    df["final_topic"] = df["final_topic"].replace({"": "Lainnya", "None": "Lainnya"})
    return df

# def find_question_variations(questions: List[str], min_variation_size: int = 3) -> List[List[str]]:
#     if not questions or len(questions) < min_variation_size * 2:
#         return [questions]

#     variations = recursive_clustering(
#         texts=questions,
#         min_cluster_size=min_variation_size,
#         max_depth=2
#     )
    
#     filtered_variations = [v for v in variations if len(v) >= min_variation_size]
    
#     if not filtered_variations:
#         return [questions]
        
#     return filtered_variations

# def generate_representative(questions: List[str]) -> str:
#     if not questions:
#         return ""

#     API_URL = "https://cloudiessky-Phi-4-mini-instruct-model.hf.space/api/predict"
    
#     headers = {"Content-Type": "application/json"}

#     sample_questions = questions[:3]
    
#     # Preprocessing untuk menghilangkan informasi sensitif
#     cleaned_questions = []
#     for q in sample_questions:
#         # Hapus nomor PO, ID transaksi, dll.
#         q_clean = re.sub(r'\bpo[a-z0-9]+\b', '[nomor pesanan]', q.lower())
#         q_clean = re.sub(r'\b[a-z0-9]{8,}\b', '[ID]', q_clean)
#         q_clean = re.sub(r'\b(kalimantan timur|jakarta|surabaya|dll)\b', '[lokasi]', q_clean)
#         q_clean = re.sub(r'\b(toko|merchant|penyedia)\s+[a-z]+\b', '[nama toko]', q_clean)
#         q_clean = re.sub(r'\bterima\s+kasih\b', '', q_clean)
#         q_clean = re.sub(r'\bmin\b|kak\b|admin\b|pak\b|bu\b', '', q_clean)
#         cleaned_questions.append(q_clean.strip())

#     prompt = f"""
# ANDA ADALAH SEORANG ANALIS LAYANAN PELANGGAN. TUGAS ANDA ADALAH MERINGKAS SEKUMPULAN PERTANYAAN PENGGUNA MENJADI SATU KALIMAT TANYA FORMAL YANG MENCERMIKAN INTI MASALAH.

# ATURAN PENTING:
# 1. HASILKAN HANYA SATU KALIMAT TANYA YANG JELAS DAN RINGKAS
# 2. JANGAN GABUNGKAN BEBERAPA PERTANYAAN DENGAN KATA "DAN" ATAU "ATAU"
# 3. HINDARI MENYERTAKAN INFORMASI SPESIFIK SEPERTI NOMOR PO, ID, LOKASI, ATAU NAMA
# 4. FOKUS PADA MASALAH UMUM YANG DIHADAPI PENGGUNA
# 5. GUNAKAN BAHASA FORMAL YANG MUDAH DIPAHAMI

# CONTOH:
# ---
# Pertanyaan Pengguna:
# - "admin solusinya bagaimana?"
# - "min ini maksudnya bagaimana ya?"

# Kalimat Tanya Representatif:
# Bagaimana cara mengatasi kendala yang sedang terjadi?
# ---

# SEKARANG, BUAT SATU KALIMAT TANYA FORMAL UNTUK PERTANYAAN-PERTANYAAN BERIKUT:
# Pertanyaan Pengguna:
# - "{cleaned_questions[0]}"
# - "{cleaned_questions[1] if len(cleaned_questions) > 1 else '...'}"
# - "{cleaned_questions[2] if len(cleaned_questions) > 2 else '...'}"

# Kalimat Tanya Representatif:
# """

#     payload = {
#         "prompt": prompt,
#         "max_new_tokens": 50,
#         "temperature": 1e-5,
#         "do_sample": False
#     }

#     try:
#         response = requests.post(API_URL, headers=headers, json=payload)
#         response.raise_for_status()
#         result = response.json()
#         representative_sentence = result["response"].strip()

#         prefixes_to_remove = ["Kalimat Tanya Representatif:", "Representatif:", "Jawaban:", "Answer:", "Pertanyaan:", "Contoh:"]
#         for pref in prefixes_to_remove:
#             if representative_sentence.lower().startswith(pref.lower()):
#                 representative_sentence = representative_sentence[len(pref):].strip()

#         representative_sentence = re.sub(r'\?[\s\-]*\?+$', '?', representative_sentence)
#         representative_sentence = re.sub(r'^[\d\.\-\*\s"]+', '', representative_sentence).strip()
#         representative_sentence = re.sub(r'\b(terima\s+kasih|mohon\s+maaf|tolong|info)\b', '', representative_sentence, flags=re.IGNORECASE)
        
#         # Pastikan hanya ada satu kalimat pertanyaan
#         if '?' in representative_sentence:
#             parts = representative_sentence.split('?')
#             if len(parts) > 1:
#                 representative_sentence = parts[0] + '?'

#         representative_sentence = re.sub(r'\s+', ' ', representative_sentence).strip()
#         rep = representative_sentence.lower().strip()
#         rep = re.sub(r'\bapakah cara\b', '', rep).strip()
#         rep = re.sub(r'\b(bagaimana cara\s+)+', 'bagaimana cara ', rep).strip()
        
#         # Logika tambahan: jika terdapat kata "cara", maka awalannya harus "Bagaimana cara"
#         if "cara" in representative_sentence.lower():
#             if not representative_sentence.lower().startswith("bagaimana cara"):
#                 representative_sentence = re.sub(
#                     r'^(bagaimana|gimana|gmna|gmn|mengapa|kenapa)\s+', 
#                     '', 
#                     representative_sentence, 
#                     flags=re.IGNORECASE
#                 ).strip()
#                 representative_sentence = "Bagaimana cara " + representative_sentence

#         if representative_sentence:
#             representative_sentence = representative_sentence[0].upper() + representative_sentence[1:]

#         if not representative_sentence.endswith('?'):
#             representative_sentence += '?'
        
#         # Jika hasil AI masih jelek, gunakan fallback cerdas
#         if len(representative_sentence) < 15 or "contoh" in representative_sentence.lower() or "pertanyaan" in representative_sentence.lower():
#             print("Hasil AI tidak memuaskan, menggunakan fallback cerdas...")
#             return smart_embedding_fallback(questions)
        
#         return representative_sentence

#     except Exception as e:
#         print(f"Error during API call: {e}. Menggunakan fallback cerdas.")
#         return smart_embedding_fallback(questions)

# def smart_embedding_fallback(questions: List[str]) -> str:
#     if not questions:
#         return ""
#     try:
#         sentence_model = get_sentence_model()

#         # Preprocessing untuk menghilangkan informasi sensitif
#         cleaned_questions = []
#         for q in questions:
#             q_clean = re.sub(r'\bpo[a-z0-9]+\b', '[nomor pesanan]', q.lower())
#             q_clean = re.sub(r'\b[a-z0-9]{8,}\b', '[ID]', q_clean)
#             q_clean = re.sub(r'\b(kalimantan timur|jakarta|surabaya|dll)\b', '[lokasi]', q_clean)
#             q_clean = re.sub(r'\b(toko|merchant|penyedia)\s+[a-z]+\b', '[nama toko]', q_clean)
#             q_clean = re.sub(r'\bterima\s+kasih\b', '', q_clean)
#             q_clean = re.sub(r'\bmin\b|kak\b|admin\b|pak\b|bu\b', '', q_clean)
#             cleaned_questions.append(q_clean.strip())

#         embeddings = sentence_model.encode(cleaned_questions, convert_to_tensor=True)
#         centroid = embeddings.mean(dim=0)
#         cosine_scores = util.cos_sim(centroid, embeddings)

#         most_similar_idx = cosine_scores.argmax().item()
#         most_representative_question = cleaned_questions[most_similar_idx]

#         rephrased = most_representative_question.strip().lower()
#         rephrased = re.sub(r'\b(gimana|gmn|bagaimana cara)\b', 'Bagaimana cara', rephrased)
#         rephrased = re.sub(r'\b(knp|kenapa)\b', 'Mengapa', rephrased)
#         rephrased = re.sub(r'\b(kak|min|admin|pak|bu)\b', '', rephrased) 
#         rephrased = re.sub(r'\s+', ' ', rephrased).strip()
        
#         # Pastikan hanya ada satu kalimat pertanyaan
#         if '?' in rephrased:
#             parts = rephrased.split('?')
#             if len(parts) > 1:
#                 rephrased = parts[0] + '?'
        
#         if rephrased:
#             rephrased = rephrased[0].upper() + rephrased[1:]
        
#         if not rephrased.endswith('?'):
#             rephrased += '?'
            
#         return rephrased

#     except Exception as e:
#         print(f"Fallback cerdas juga gagal: {e}. Menggunakan fallback generik.")
#         return "Apa solusi untuk masalah yang dialami?"

def find_question_variations(questions: List[str], min_variation_size: int = 3) -> List[List[str]]:
    if not questions or len(questions) < min_variation_size * 2:
        return [questions]

    # Perbaikan clustering dengan threshold yang lebih baik
    variations = improved_recursive_clustering(
        texts=questions,
        min_cluster_size=min_variation_size,
        max_depth=2
    )
    
    filtered_variations = [v for v in variations if len(v) >= min_variation_size]
    
    if not filtered_variations:
        return [questions]
        
    return filtered_variations

def improved_recursive_clustering(texts: List[str], min_cluster_size: int, max_depth: int, current_depth: int = 0) -> List[List[str]]:
    if current_depth >= max_depth or len(texts) < min_cluster_size * 2:
        return [texts]
    
    # Preprocessing yang lebih agresif untuk clustering
    cleaned_texts = []
    for text in texts:
        # Normalisasi teks untuk clustering
        clean_text = re.sub(r'\bpo[a-z0-9]+\b', '[id]', text.lower())
        clean_text = re.sub(r'\b[a-z0-9]{8,}\b', '[id]', clean_text)
        clean_text = re.sub(r'\b(min|kak|admin|pak|bu|dear)\b', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        cleaned_texts.append(clean_text)
    
    try:
        # Gunakan sentence transformer untuk embedding
        sentence_model = get_sentence_model()
        embeddings = sentence_model.encode(cleaned_texts, convert_to_tensor=True)
        
        # Hitung similarity matrix
        from sentence_transformers import util
        cosine_scores = util.cos_sim(embeddings, embeddings)
        
        # Clustering dengan threshold yang disesuaikan
        from sklearn.cluster import AgglomerativeClustering
        import numpy as np
        
        # Konversi similarity ke distance
        distance_matrix = 1 - cosine_scores.cpu().numpy()
        
        # Threshold dinamis berdasarkan jumlah data
        threshold = 0.7 if len(texts) > 10 else 0.8
        
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            linkage='average',
            metric='precomputed'
        )
        
        cluster_labels = clustering_model.fit_predict(distance_matrix)
        
        # Kelompokkan berdasarkan label
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[i])  # Gunakan teks asli, bukan yang sudah dibersihkan
        
        # Filter cluster yang memenuhi ukuran minimum
        valid_clusters = [cluster for cluster in clusters.values() 
                         if len(cluster) >= min_cluster_size]
        
        # Jika tidak ada cluster yang valid, kembalikan semua teks
        if not valid_clusters:
            return [texts]
        
        # Jika masih dalam kedalaman maksimal, lanjutkan clustering rekursif
        if current_depth < max_depth - 1:
            final_clusters = []
            for cluster in valid_clusters:
                if len(cluster) >= min_cluster_size * 2:
                    # Lanjutkan clustering untuk cluster besar
                    sub_clusters = improved_recursive_clustering(
                        texts=cluster,
                        min_cluster_size=min_cluster_size,
                        max_depth=max_depth,
                        current_depth=current_depth + 1
                    )
                    final_clusters.extend(sub_clusters)
                else:
                    final_clusters.append(cluster)
            return final_clusters
        else:
            return valid_clusters
            
    except Exception as e:
        print(f"Error in clustering: {e}")
        # Fallback: jika clustering gagal, bagi rata
        if len(texts) >= min_cluster_size * 2:
            mid = len(texts) // 2
            return [texts[:mid], texts[mid:]]
        else:
            return [texts]

def generate_representative(questions: List[str]) -> str:
    if not questions:
        return ""

    API_URL = "https://cloudiessky-Phi-4-mini-instruct-model.hf.space/api/predict"
    
    headers = {"Content-Type": "application/json"}

    sample_questions = questions[:3]
    
    # Preprocessing untuk menghilangkan informasi sensitif - LEBIH HATI-HATI
    cleaned_questions = []
    for q in sample_questions:
        # Hapus nomor PO dengan placeholder yang KONSISTEN
        q_clean = re.sub(r'\bpo[a-z0-9]{10,}\b', '[NOMOR_PO]', q.lower())
        # Hapus ID transaksi/random
        q_clean = re.sub(r'\b[a-z0-9]{8,}\b', '[ID]', q_clean)
        # Hapus lokasi spesifik
        q_clean = re.sub(r'\b(kalimantan timur|jakarta|surabaya|bandung|medan|makassar|sulawesi selatan)\b', '[LOKASI]', q_clean)
        # Hapus nama toko spesifik
        q_clean = re.sub(r'\b(toko|merchant|penyedia|cv|pt)\s+[a-z]+\b', '[NAMA_TOKO]', q_clean)
        # Hapus frasa umum yang tidak perlu
        q_clean = re.sub(r'\b(terima\s+kasih|terimakasih|mohon\s+bantu|bantu|petunjuk|info|maaf)\b', '', q_clean, flags=re.IGNORECASE)
        # Hapus panggilan admin
        q_clean = re.sub(r'\b(min|kak|admin|pak|bu|om|tante|cs|customer service)\b', '', q_clean, flags=re.IGNORECASE)
        # Bersihkan spasi berlebih
        q_clean = re.sub(r'\s+', ' ', q_clean).strip()
        cleaned_questions.append(q_clean)

    prompt = f"""
ANDA ADALAH SEORANG ANALIS LAYANAN PELANGGAN. TUGAS ANDA ADALAH MERINGKAS SEKUMPULAN PERTANYAAN PENGGUNA MENJADI SATU KALIMAT TANYA FORMAL YANG MENCERMIKAN INTI MASALAH.

ATURAN PENTING:
1. HASILKAN HANYA SATU KALIMAT TANYA YANG JELAS DAN RINGKAS
2. JANGAN GABUNGKAN BEBERAPA PERTANYAAN DENGAN KATA "DAN" ATAU "ATAU"
3. HINDARI MENYERTAKAN INFORMASI SPESIFIK SEPERTI NOMOR PO, ID, LOKASI, ATAU NAMA
4. FOKUS PADA MASALAH UMUM YANG DIHADAPI PENGGUNA
5. GUNAKAN BAHASA FORMAL YANG MUDAH DIPAHAMI

CONTOH:
---
Pertanyaan Pengguna:
- "admin solusinya bagaimana?"
- "min ini maksudnya bagaimana ya?"

Kalimat Tanya Representatif:
Bagaimana cara mengatasi kendala yang sedang terjadi?
---

SEKARANG, BUAT SATU KALIMAT TANYA FORMAL UNTUK PERTANYAAN-PERTANYAAN BERIKUT:
Pertanyaan Pengguna:
- "{cleaned_questions[0]}"
- "{cleaned_questions[1] if len(cleaned_questions) > 1 else '...'}"
- "{cleaned_questions[2] if len(cleaned_questions) > 2 else '...'}"

Kalimat Tanya Representatif:
"""

    payload = {
        "prompt": prompt,
        "max_new_tokens": 50,
        "temperature": 1e-5,
        "do_sample": False
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        representative_sentence = result["response"].strip()

        # Post-processing yang lebih ketat
        representative_sentence = clean_representative_sentence(representative_sentence)
        
        # Validasi hasil AI - hanya fallback jika benar-benar tidak memadai
        if is_valid_representative(representative_sentence):
            return representative_sentence
        else:
            print("Hasil AI tidak memenuhi kriteria, menggunakan fallback cerdas...")
            return smart_embedding_fallback(questions)

    except Exception as e:
        print(f"Error during API call: {e}. Menggunakan fallback cerdas.")
        return smart_embedding_fallback(questions)

# def clean_representative_sentence(sentence: str) -> str:
#     """Membersihkan kalimat representatif dengan post-processing yang lebih baik"""
#     if not sentence:
#         return ""
    
#     # Hapus prefix yang tidak diinginkan
#     prefixes_to_remove = [
#         "Kalimat Tanya Representatif:", "Representatif:", "Jawaban:", 
#         "Answer:", "Pertanyaan:", "Contoh:", "Kalimat Tanya:"
#     ]
#     for pref in prefixes_to_remove:
#         if sentence.lower().startswith(pref.lower()):
#             sentence = sentence[len(pref):].strip()

#     # Bersihkan karakter berlebih
#     sentence = re.sub(r'\?[\s\-]*\?+$', '?', sentence)
#     sentence = re.sub(r'^[\d\.\-\*\s"]+', '', sentence).strip()
#     sentence = re.sub(r'\b(terima\s+kasih|mohon\s+maaf|tolong|info)\b', '', sentence, flags=re.IGNORECASE)
    
#     # Pastikan hanya ada satu kalimat pertanyaan
#     if '?' in sentence:
#         parts = sentence.split('?')
#         sentence = parts[0] + '?'

#     sentence = re.sub(r'\s+', ' ', sentence).strip()
    
#     # Normalisasi awalan pertanyaan
#     sentence_lower = sentence.lower().strip()
#     if "cara" in sentence_lower and not sentence_lower.startswith("bagaimana cara"):
#         sentence = re.sub(r'^(bagaimana|gimana|gmna|gmn|mengapa|kenapa)\s+', '', sentence, flags=re.IGNORECASE).strip()
#         sentence = "Bagaimana cara " + sentence

#     # Format akhir
#     if sentence:
#         sentence = sentence[0].upper() + sentence[1:]
    
#     if sentence and not sentence.endswith('?'):
#         sentence += '?'
        
#     return sentence.strip()

def clean_representative_sentence(sentence: str) -> str:
    if not sentence:
        return ""
    
    # Hapus prefix yang tidak diinginkan
    prefixes_to_remove = [
        "Kalimat Tanya Representatif:", "Representatif:", "Jawaban:", 
        "Answer:", "Pertanyaan:", "Contoh:", "Kalimat Tanya:", "Response:",
        "hasil:", "output:", "Berikut"
    ]
    sentence_lower = sentence.lower().strip()
    for pref in prefixes_to_remove:
        if sentence_lower.startswith(pref.lower()):
            sentence = sentence[len(pref):].strip()
            break

    # Bersihkan karakter berlebih
    sentence = re.sub(r'\?[\s\-]*\?+$', '?', sentence)
    sentence = re.sub(r'^[\d\.\-\*\s"]+', '', sentence).strip()
    
    # Hapus placeholder yang tidak valid
    sentence = re.sub(r'\[.*?\]', '', sentence)  # Hapus semua placeholder dulu
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    # Pastikan hanya ada satu kalimat pertanyaan
    if '?' in sentence:
        parts = sentence.split('?')
        sentence = parts[0] + '?'

    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    # Normalisasi awalan pertanyaan
    sentence_clean = sentence.lower().strip()
    if sentence_clean and not sentence_clean.startswith(("bagaimana", "apa", "mengapa", "kenapa", "kapan", "cara", "solusi", "penyebab")):
        if any(word in sentence_clean for word in ["error", "gangguan", "masalah", "kendala"]):
            sentence = "Bagaimana cara mengatasi kendala teknis?"
        elif any(word in sentence_clean for word in ["bayar", "pembayaran", "transfer", "dana", "cair"]):
            sentence = "Bagaimana cara menyelesaikan masalah pembayaran?"
        elif any(word in sentence_clean for word in ["login", "akses", "masuk"]):
            sentence = "Apa solusi untuk masalah akses akun?"
        else:
            sentence = "Bagaimana cara " + sentence_clean
    
    # Format akhir
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    
    if sentence and not sentence.endswith('?'):
        sentence += '?'
        
    return sentence.strip()

# def is_valid_representative(sentence: str) -> bool:
#     """Validasi apakah kalimat representatif memenuhi kriteria minimum"""
#     if not sentence or len(sentence.strip()) < 10:
#         return False
    
#     # Hindari fallback dini - hanya jika benar-benar tidak valid
#     invalid_indicators = [
#         "contoh", "pertanyaan", "jawaban", "representatif", 
#         "kalimat tanya", "berikut", "sebagai"
#     ]
    
#     sentence_lower = sentence.lower()
#     if any(indicator in sentence_lower for indicator in invalid_indicators):
#         return False
    
#     # Harus mengandung kata tanya dan memiliki struktur pertanyaan
#     if '?' not in sentence:
#         return False
    
#     # Minimal panjang yang masuk akal
#     if len(sentence) < 15 or len(sentence) > 150:
#         return False
    
#     # Harus mengandung kata kerja atau kata kunci pertanyaan
#     question_keywords = ["bagaimana", "apa", "mengapa", "kenapa", "kapan", "cara", "solusi"]
#     if not any(keyword in sentence_lower for keyword in question_keywords):
#         return False
    
#     return True

def is_valid_representative(sentence: str) -> bool:
    if not sentence or len(sentence.strip()) < 20:
        return False
    
    # Cek placeholder yang tidak diizinkan
    forbidden_placeholders = ['[id]', '[nomor pesanan]', '[po]', 'po68', '[noid]', '[no]']
    sentence_lower = sentence.lower()
    if any(ph in sentence_lower for ph in forbidden_placeholders):
        return False
    
    # Harus mengandung kata tanya
    if '?' not in sentence:
        return False
    
    # Panjang yang masuk akal
    if len(sentence) < 25 or len(sentence) > 100:
        return False
    
    # Harus mengandung kata kunci pertanyaan
    question_keywords = ["bagaimana", "apa", "mengapa", "kenapa", "kapan", "cara", "solusi", "penyebab", "masalah"]
    if not any(keyword in sentence_lower for keyword in question_keywords):
        return False
    
    # Hindari kalimat terlalu umum
    too_generic = ["apa saja", "bagaimana ya", "kenapa ya", "gimana ya", "tolong bantu"]
    if any(generic in sentence_lower for generic in too_generic):
        return False
    
    # Harus terdengar profesional
    if sentence.count('...') > 0 or sentence.count('??') > 0:
        return False
    
    return True

# def smart_embedding_fallback(questions: List[str]) -> str:
#     """Fallback yang hanya digunakan ketika benar-benar diperlukan"""
#     if not questions:
#         return "Apa solusi untuk masalah yang dialami?"
    
#     try:
#         sentence_model = get_sentence_model()

#         # Preprocessing untuk menghilangkan informasi sensitif
#         cleaned_questions = []
#         for q in questions:
#             q_clean = re.sub(r'\bpo[a-z0-9]+\b', '[nomor pesanan]', q.lower())
#             q_clean = re.sub(r'\b[a-z0-9]{8,}\b', '[ID]', q_clean)
#             q_clean = re.sub(r'\b(kalimantan timur|jakarta|surabaya|dll)\b', '[lokasi]', q_clean)
#             q_clean = re.sub(r'\b(toko|merchant|penyedia)\s+[a-z]+\b', '[nama toko]', q_clean)
#             q_clean = re.sub(r'\bterima\s+kasih\b', '', q_clean)
#             q_clean = re.sub(r'\bmin\b|kak\b|admin\b|pak\b|bu\b', '', q_clean)
#             cleaned_questions.append(q_clean.strip())

#         # Gunakan embedding untuk mencari pertanyaan paling representatif
#         embeddings = sentence_model.encode(cleaned_questions, convert_to_tensor=True)
#         centroid = embeddings.mean(dim=0)
#         cosine_scores = util.cos_sim(centroid, embeddings)

#         most_similar_idx = cosine_scores.argmax().item()
#         most_representative_question = cleaned_questions[most_similar_idx]

#         # Transformasi ke kalimat tanya representatif yang lebih baik
#         rephrased = most_representative_question.strip().lower()
        
#         # Pattern matching untuk transformasi yang lebih baik
#         if "error" in rephrased or "gangguan" in rephrased:
#             return "Apa penyebab kendala teknis yang sedang terjadi?"
#         elif "bayar" in rephrased or "pembayaran" in rephrased:
#             return "Bagaimana cara menyelesaikan masalah pembayaran?"
#         elif "login" in rephrased or "akses" in rephrased:
#             return "Apa solusi untuk masalah akses akun?"
#         elif "verifikasi" in rephrased:
#             return "Berapa lama waktu verifikasi pembayaran?"
#         elif "nib" in rephrased:
#             return "Apa syarat pembuatan NIB untuk toko?"
#         elif "notif" in rephrased or "wa" in rephrased:
#             return "Bagaimana cara mengatur notifikasi pesan?"
#         elif "upload" in rephrased or "produk" in rephrased:
#             return "Bagaimana cara mengatasi masalah upload produk?"
#         elif "bast" in rephrased:
#             return "Bagaimana cara memperbaiki kesalahan input BAST?"
#         else:
#             # Transformasi umum
#             rephrased = re.sub(r'\b(gimana|gmn|bagaimana cara)\b', 'Bagaimana cara', rephrased)
#             rephrased = re.sub(r'\b(knp|kenapa)\b', 'Mengapa', rephrased)
#             rephrased = re.sub(r'\b(kak|min|admin|pak|bu)\b', '', rephrased) 
#             rephrased = re.sub(r'\s+', ' ', rephrased).strip()
            
#             # Pastikan hanya ada satu kalimat pertanyaan
#             if '?' in rephrased:
#                 parts = rephrased.split('?')
#                 if len(parts) > 1:
#                     rephrased = parts[0] + '?'
            
#             if rephrased:
#                 rephrased = rephrased[0].upper() + rephrased[1:]
            
#             if not rephrased.endswith('?'):
#                 rephrased += '?'
                
#             # Validasi akhir
#             if len(rephrased) > 10 and '?' in rephrased:
#                 return rephrased
#             else:
#                 return "Apa solusi untuk masalah yang dialami?"

#     except Exception as e:
#         print(f"Fallback cerdas juga gagal: {e}. Menggunakan fallback generik.")
#         return "Apa solusi untuk masalah yang dialami?"

def smart_embedding_fallback(questions: List[str]) -> str:
    """Fallback yang hanya digunakan ketika benar-benar diperlukan"""
    if not questions:
        return "Apa solusi untuk masalah yang dialami?"
    
    try:
        sentence_model = get_sentence_model()

        # Preprocessing untuk menghilangkan informasi sensitif
        cleaned_questions = []
        for q in questions:
            q_clean = re.sub(r'\bpo[a-z0-9]+\b', '[nomor pesanan]', q.lower())
            q_clean = re.sub(r'\b[a-z0-9]{8,}\b', '[ID]', q_clean)
            q_clean = re.sub(r'\b(kalimantan timur|jakarta|surabaya|dll|bandung|medan)\b', '[lokasi]', q_clean)
            q_clean = re.sub(r'\b(toko|merchant|penyedia|cv|pt)\s+[a-z]+\b', '[nama toko]', q_clean)
            q_clean = re.sub(r'\bterima\s+kasih\b', '', q_clean)
            q_clean = re.sub(r'\b(min|kak|admin|pak|bu|om|tante)\b', '', q_clean)
            q_clean = re.sub(r'\s+', ' ', q_clean).strip()
            cleaned_questions.append(q_clean)

        # Gunakan embedding untuk mencari pertanyaan paling representatif
        embeddings = sentence_model.encode(cleaned_questions, convert_to_tensor=True)
        centroid = embeddings.mean(dim=0)
        cosine_scores = util.cos_sim(centroid, embeddings)

        most_similar_idx = cosine_scores.argmax().item()
        most_representative_question = cleaned_questions[most_similar_idx]

        # Transformasi ke kalimat tanya representatif yang lebih baik
        rephrased = most_representative_question.strip().lower()
        
        # Pattern matching untuk transformasi yang lebih baik - lebih umum
        if any(word in rephrased for word in ["error", "gangguan", "masalah", "kendala", "eror", "trobel"]):
            return "Bagaimana cara mengatasi kendala teknis yang sedang terjadi?"
        elif any(word in rephrased for word in ["bayar", "pembayaran", "transfer", "dana", "cair"]):
            return "Bagaimana cara menyelesaikan masalah pembayaran?"
        elif any(word in rephrased for word in ["login", "akses", "masuk", "bisa"]):
            return "Apa solusi untuk masalah akses akun?"
        elif any(word in rephrased for word in ["verifikasi", "otp", "otentikasi", "auth"]):
            return "Bagaimana cara menyelesaikan masalah verifikasi?"
        elif any(word in rephrased for word in ["nib", "npwp", "dokumen", "legal"]):
            return "Apa syarat dan prosedur untuk dokumen legal usaha?"
        elif any(word in rephrased for word in ["notif", "wa", "email", "notifikasi"]):
            return "Bagaimana cara mengatur notifikasi pesan?"
        elif any(word in rephrased for word in ["upload", "produk", "unggah", "foto"]):
            return "Bagaimana cara mengatasi masalah upload produk?"
        elif any(word in rephrased for word in ["bast", "dokumen", "ttd", "tte"]):
            return "Bagaimana cara memperbaiki kesalahan input dokumen?"
        elif any(word in rephrased for word in ["rekening", "bank", "transfer"]):
            return "Apa syarat dan prosedur untuk mengganti rekening?"
        elif any(word in rephrased for word in ["modal", "talangan", "usaha"]):
            return "Bagaimana cara mengajukan bantuan modal usaha?"
        elif any(word in rephrased for word in ["pajak", "ppn", "npwp", "coretax"]):
            return "Bagaimana cara menangani masalah perpajakan?"
        elif any(word in rephrased for word in ["pengiriman", "kurir", "barang"]):
            return "Bagaimana cara menyelesaikan masalah pengiriman barang?"
        else:
            # Transformasi umum
            rephrased = re.sub(r'\b(gimana|gmn|bagaimana cara)\b', 'Bagaimana cara', rephrased)
            rephrased = re.sub(r'\b(knp|kenapa)\b', 'Mengapa', rephrased)
            rephrased = re.sub(r'\b(kak|min|admin|pak|bu|om|tante)\b', '', rephrased) 
            rephrased = re.sub(r'\s+', ' ', rephrased).strip()
            
            # Pastikan hanya ada satu kalimat pertanyaan
            if '?' in rephrased:
                parts = rephrased.split('?')
                rephrased = parts[0] + '?'
            
            if rephrased:
                rephrased = rephrased[0].upper() + rephrased[1:]
            
            if not rephrased.endswith('?'):
                rephrased += '?'
                
            # Validasi akhir
            if len(rephrased) > 15 and '?' in rephrased and len(rephrased) < 120:
                return rephrased
            else:
                return "Apa solusi untuk masalah yang dialami?"

    except Exception as e:
        print(f"Fallback cerdas juga gagal: {e}. Menggunakan fallback generik.")
        return "Apa solusi untuk masalah yang dialami?"

# Run
if __name__ == '__main__':
    sample_data = {
        'text': [
            'Bagaimana cara verifikasi toko saya?',
            'Dana saya belum masuk rekening, tolong dicek.',
            'Kapan pencairan dana gelombang 2?',
            'Saya tidak bisa login ke aplikasi, ada masalah apa?',
            'Bagaimana cara upload produk massal?',
            'Ada kendala akses web, tidak bisa dibuka.',
            'Apakah ada info terbaru tentang pajak PPN?',
            'Saya ingin bertanya tentang etika penggunaan platform.',
            'Pembayaran saya pending, mohon dibantu.',
            'Barang yang dikirim rusak, bagaimana ini?',
            'Ini topik baru yang belum ada di list keyword sama sekali.',
            'Pesan ini juga tentang topik baru yang mirip dengan yang sebelumnya.',
            'Ini adalah pesan yang sangat berbeda dan harusnya jadi topik baru lagi.',
            'Saya butuh bantuan umum, tidak spesifik.',
            'Verifikasi pembayaran saya gagal, bagaimana solusinya?',
            'Tanda tangan elektronik saya tidak berfungsi.',
            'Bagaimana cara mengubah data toko?',
            'Pengajuan modal saya dibatalkan, kenapa ya?',
            'Ada masalah dengan autentikasi OTP.',
            'Saya tidak bisa mengunggah gambar produk.',
            'Kapan kurir akan menjemput barang?',
            'Bagaimana cara menggunakan fitur siplah?',
            'Status pesanan saya masih menggantung.',
            'Ada pertanyaan umum lainnya.'
        ]
    }
    df_sample = pd.DataFrame(sample_data)
    topik_keywords_example = {
        "verifikasi_toko": [["verifikasi", "toko"]],
        "dana_belum_masuk": [["dana", "belum", "masuk"]],
        "jadwal_cair_dana": [["kapan", "cair"], ["kapan", "pencairan"]],
        "kendala_akses": [["tidak", "bisa", "login"], ["kendala", "akses"]],
        "kendala_upload": [["upload", "produk"], ["unggah", "gambar"]],
        "pajak": [["pajak", "ppn"], ["ppn"]],
        "etika_penggunaan": [["etika", "penggunaan"]],
        "pembayaran_dana": ["pembayaran", "pending"],
        "pengiriman_barang": ["barang", "rusak"],
        "bantuan_umum": ["bantuan", "umum", "tanya"]
    }
    spelling = load_spelling_corrections('kata_baku.csv')

    print('Running optimized clustering + keyword matching...')
    result = integrate_clustering_with_keywords(df_sample.copy(), topik_keywords_example, spelling_corrections=spelling)

    df_result, new_topics, num_new = result
    print("\n=== Hasil Integrasi Clustering + Keyword ===")
    print(df_result.head())

    df_merged = merge_similar_topics(df_result, use_embeddings=True)
    print("\n=== Setelah Merge Similar Topics ===")
    print(df_merged['final_topic'].value_counts())


