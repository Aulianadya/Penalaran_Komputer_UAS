
    import os
    import json
    import re
    import torch
    import pandas as pd
    import numpy as np
    from typing import List
    from collections import Counter
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import AutoTokenizer, AutoModel
    from sklearn.feature_extraction.text import TfidfVectorizer

    # --- Load Data ---
    # Asumsi sudah tersedia:
    # texts, case_ids, tfidf_vectors, bert_vectors, X_train_tfidf, X_train_bert
    # ids_train_tfidf, ids_train_bert

    # --- TF-IDF Vectorizer (dari tahap sebelumnya) ---
    # vectorizer_tfidf = TfidfVectorizer(...) # sudah fit dan transform sebelumnya

    # --- Load Model BERT ---
    model_name = "indobenchmark/indobert-base-p1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # --- Clean Text ---
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()

    # --- Fungsi Retrieval ---
    def retrieve(query: str, k: int = 5, method: str = "tfidf") -> List[str]:
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

    # --- Ekstrak Solusi dari Amar Putusan ---
    def load_case_solutions(txt_folder: str) -> dict:
        case_solutions = {}
        file_paths = [os.path.join(txt_folder, f) for f in os.listdir(txt_folder) if f.endswith("_clean.txt")]
        for path in file_paths:
            case_id = os.path.basename(path).replace("_clean.txt", "")
            with open(path, encoding="utf-8") as f:
                text = f.read()
            text_lower = text.lower()
            match = re.search(r"amar putusan\s*([\s\S]{50,2000}?)

", text_lower)
            if match:
                solution_text = match.group(1).strip()
            else:
                solution_text = text[-300:].strip()
            case_solutions[case_id] = solution_text

        print(f"Jumlah kasus dengan solusi berhasil diekstrak: {len(case_solutions)}")
        return case_solutions

    # --- Prediksi Solusi Berdasarkan Voting atau Weighted Similarity ---
    def predict_outcome(query: str, case_solutions: dict, method="tfidf", use_weight=False) -> str:
        top_k_ids = retrieve(query, k=5, method=method)

        if use_weight:
            query_clean = clean_text(query)
            if method == "tfidf":
                query_vec = vectorizer_tfidf.transform([query_clean])
                similarities = cosine_similarity(query_vec, X_train_tfidf).flatten()
                sim_scores = [similarities[ids_train_tfidf.index(cid)] for cid in top_k_ids]
            else:
                inputs = tokenizer(query_clean, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    query_vec = outputs.last_hidden_state[:, 0, :].numpy()
                similarities = cosine_similarity(query_vec, X_train_bert).flatten()
                sim_scores = [similarities[ids_train_bert.index(cid)] for cid in top_k_ids]

            weighted_solutions = {}
            for cid, sim in zip(top_k_ids, sim_scores):
                sol = case_solutions.get(cid, "")
                weighted_solutions[sol] = weighted_solutions.get(sol, 0) + sim
            predicted_solution = max(weighted_solutions, key=weighted_solutions.get)
        else:
            solutions = [case_solutions.get(cid, "") for cid in top_k_ids]
            most_common = Counter(solutions).most_common(1)
            predicted_solution = most_common[0][0] if most_common else ""

        return predicted_solution

    # --- Evaluasi Manual dan Simpan ---
    def run_prediction_and_save(eval_file_path: str, case_solutions: dict, method="tfidf"):
        with open(eval_file_path, "r", encoding="utf-8") as f:
            eval_queries = json.load(f)

        results = []
        for i, q in enumerate(eval_queries):
            pred = predict_outcome(q["query"], case_solutions, method=method, use_weight=True)
            top5 = retrieve(q["query"], k=5, method=method)
            results.append({
                "query_id": f"Q{i+1}",
                "predicted_solution": pred,
                "top_5_case_ids": top5
            })
            print(f"[Q{i+1}]")
            print("Prediksi Solusi  :", pred[:200])
            print("Top-5 Case IDs   :", top5)
            print("-" * 50)

        df = pd.DataFrame(results)
        output_dir = "/content/drive/MyDrive/UASPK/data/results"
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
        print("Disimpan di:", output_dir + "/predictions.csv")
