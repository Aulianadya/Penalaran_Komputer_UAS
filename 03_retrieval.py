
    import os
    import re
    import glob
    import json
    import random
    import numpy as np
    import pandas as pd
    from typing import List
    from tqdm import tqdm
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import AutoTokenizer, AutoModel
    import torch

    # --- Tahap 3: Case Retrieval ---

    # (i) Representasi Vektor

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    folder_txt = "/content/drive/MyDrive/UASPK/TXT"
    file_paths = sorted(glob.glob(os.path.join(folder_txt, "*.txt")))

    texts, case_ids = [], []
    for path in file_paths:
        with open(path, encoding="utf-8") as f:
            raw_text = f.read()
        texts.append(clean_text(raw_text))
        case_ids.append(os.path.basename(path).replace("_clean.txt", ""))

    # TF-IDF
    vectorizer_tfidf = TfidfVectorizer(max_features=5000)
    tfidf_vectors = vectorizer_tfidf.fit_transform(texts)
    print("TF-IDF vector shape:", tfidf_vectors.shape)

    # BERT Embedding
    model_name = "indobenchmark/indobert-base-p1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    def bert_embed(texts):
        embeddings = []
        with torch.no_grad():
            for text in tqdm(texts):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embedding.squeeze().numpy())
        return np.vstack(embeddings)

    bert_vectors = bert_embed(texts)
    print("BERT embedding shape:", bert_vectors.shape)

    # (ii) Splitting Data
    labels = [0] * len(texts)

    X_train_tfidf, X_test_tfidf, y_train, y_test, ids_train_tfidf, ids_test_tfidf = train_test_split(
        tfidf_vectors, labels, case_ids, test_size=0.2, random_state=42
    )
    print("
=== TF-IDF SPLIT ===")
    print("Jumlah data TF-IDF train:", X_train_tfidf.shape[0])
    print("Jumlah data TF-IDF test :", X_test_tfidf.shape[0])
    print("Contoh ID train:", ids_train_tfidf[:3])
    print("Contoh ID test :", ids_test_tfidf[:3])

    X_train_bert, X_test_bert, _, _, ids_train_bert, ids_test_bert = train_test_split(
        bert_vectors, labels, case_ids, test_size=0.2, random_state=42
    )
    print("
=== BERT SPLIT ===")
    print("Jumlah data BERT train:", X_train_bert.shape[0])
    print("Jumlah data BERT test :", X_test_bert.shape[0])
    print("Contoh ID train:", ids_train_bert[:3])
    print("Contoh ID test :", ids_test_bert[:3])

    # (iii) Model Retrieval

    def retrieve_tfidf(query: str, k: int = 10):
        query_clean = clean_text(query)
        query_vec = vectorizer_tfidf.transform([query_clean])
        similarities = cosine_similarity(query_vec, X_train_tfidf).flatten()
        top_k_idx = similarities.argsort()[::-1][:k]
        return [ids_train_tfidf[i] for i in top_k_idx]

    def retrieve_bert(query: str, k: int = 10):
        query_clean = clean_text(query)
        inputs = tokenizer(query_clean, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            query_vec = outputs.last_hidden_state[:, 0, :].numpy()
        similarities = cosine_similarity(query_vec, X_train_bert).flatten()
        top_k_idx = similarities.argsort()[::-1][:k]
        return [ids_train_bert[i] for i in top_k_idx]

    query_example = "Tindak pidana penganiayaan berat terhadap korban hingga luka parah"
    print("
Hasil Retrieval TF-IDF:")
    print(retrieve_tfidf(query_example))

    print("
Hasil Retrieval BERT:")
    print(retrieve_bert(query_example))

    # (iv) Fungsi General

    def retrieve(query: str, k: int = 10, method: str = "tfidf") -> List[str]:
        query_clean = clean_text(query)
        if method == "tfidf":
            query_vec = vectorizer_tfidf.transform([query_clean])
            similarities = cosine_similarity(query_vec, X_train_tfidf).flatten()
            top_k_idx = similarities.argsort()[::-1][:k]
            return [ids_train_tfidf[i] for i in top_k_idx]
        elif method == "bert":
            inputs = tokenizer(query_clean, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                query_vec = outputs.last_hidden_state[:, 0, :].numpy()
            similarities = cosine_similarity(query_vec, X_train_bert).flatten()
            top_k_idx = similarities.argsort()[::-1][:k]
            return [ids_train_bert[i] for i in top_k_idx]
        else:
            raise ValueError("Method harus 'tfidf' atau 'bert'")

    top_cases = retrieve("Penganiayaan terhadap anak di bawah umur", k=5, method="tfidf")
    print("
Top-5 kasus mirip:", top_cases)

    # (v) Pengujian Awal

    random.seed(42)
    eval_dir = "/content/drive/MyDrive/UASPK/data/eval"
    os.makedirs(eval_dir, exist_ok=True)

    num_queries = 10
    selected_indices = random.sample(range(len(texts)), num_queries)

    queries_data = []
    for i in selected_indices:
        query_text = texts[i]
        query_id = case_ids[i]
        query_vec = tfidf_vectors[i]
        similarities = cosine_similarity(query_vec, tfidf_vectors).flatten()
        sorted_idx = similarities.argsort()[::-1]
        second_best_idx = sorted_idx[1]
        ground_truth_id = case_ids[second_best_idx]
        queries_data.append({"query": query_text, "ground_truth": ground_truth_id})

    with open(os.path.join(eval_dir, "queries.json"), "w", encoding="utf-8") as f:
        json.dump(queries_data, f, indent=4, ensure_ascii=False)
    print("
File queries.json berhasil disimpan di:", eval_dir)

    # Tampilkan isi file
    print("
Isi queries.json:")
    for i, q in enumerate(queries_data, start=1):
        print(f"[{i}] Query     : {q['query'][:80]}...")
        print(f"    Ground ID : {q['ground_truth']}
")

    # Format ke DataFrame
    df_queries = pd.DataFrame(queries_data)
    print("
Preview DataFrame queries:")
    print(df_queries.head())
