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

# === Paths ===
base_path = "./"
csv_file = os.path.join(base_path, "data", "putusan_ma__2025-06-07.csv")
eval_file = os.path.join(base_path, "data/eval", "queries.json")
result_folder = os.path.join(base_path, "data/results")
os.makedirs(result_folder, exist_ok=True)

# === Load BERT Model ===
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# === Clean Text ===
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# === Load Case Solutions from CSV ===
df = pd.read_csv(csv_file)
df = df.dropna(subset=["nomor", "catatan_amar"])
df["nomor"] = df["nomor"].astype(str)
df["catatan_amar"] = df["catatan_amar"].astype(str).apply(clean_text)

case_solutions = dict(zip(df["nomor"], df["catatan_amar"]))
X_train_texts = list(case_solutions.values())
ids_train_tfidf = list(case_solutions.keys())

# === TF-IDF Vectorizer ===
vectorizer_tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train_texts)

# === Retrieve Top-k Similar Case IDs ===
def retrieve(query: str, k: int = 5, method: str = "tfidf") -> List[str]:
    query_clean = clean_text(query)

    if method == "tfidf":
        query_vec = vectorizer_tfidf.transform([query_clean])
        similarities = cosine_similarity(query_vec, X_train_tfidf).flatten()
        top_k_idx = similarities.argsort()[::-1][:k]
        return [ids_train_tfidf[i] for i in top_k_idx]

    else:
        raise ValueError("Method harus 'tfidf' (bert belum diaktifkan di versi ini)")

# === Predict Outcome ===
def predict_outcome(query: str, case_solutions: dict, method="tfidf", use_weight=False) -> str:
    top_k_ids = retrieve(query, k=5, method=method)

    if use_weight:
        query_clean = clean_text(query)
        query_vec = vectorizer_tfidf.transform([query_clean])
        similarities = cosine_similarity(query_vec, X_train_tfidf).flatten()
        sim_scores = [similarities[ids_train_tfidf.index(cid)] for cid in top_k_ids]

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

# === Run Prediction and Save CSV ===
def run_prediction_and_save(eval_file_path: str, case_solutions: dict, method="tfidf"):
    with open(eval_file_path, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    results = []
    for i, q in enumerate(eval_queries):
        query_text = q["query"]
        ground_truth_id = q["ground_truth"]
        actual_solution = case_solutions.get(ground_truth_id, "").strip()

        predicted_solution = predict_outcome(query_text, case_solutions, method=method, use_weight=True)
        top5 = retrieve(query_text, k=5, method=method)
        match = predicted_solution.strip() == actual_solution

        results.append({
            "query_id": f"Q{i+1}",
            "query": query_text,
            "predicted_solution": predicted_solution,
            "actual_solution": actual_solution,
            "top_5_case_ids": top5,
        })

        print(f"[Q{i+1}]")
        print("Query              :", query_text[:80], "...")
        print("Prediksi Solusi    :", predicted_solution[:150], "...")
        print("Actual Solution    :", actual_solution[:150], "...")
        print("Top-5 Case IDs     :", top5)
        print("-" * 50)

    df = pd.DataFrame(results)
    out_csv = os.path.join(result_folder, "predictions.csv")
    df.to_csv(out_csv, index=False)
    print("✅ Disimpan di:", out_csv)

# === Run ===
run_prediction_and_save(eval_file, case_solutions, method="tfidf")
