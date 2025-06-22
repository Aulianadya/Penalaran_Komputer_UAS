
    import os
    import json
    import re
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # --- Load Queries JSON ---
    eval_path = "/content/drive/MyDrive/UASPK/data/eval/queries.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    # --- Fungsi Retrieval (harus sudah didefinisikan sebelumnya) ---
    # retrieve(query: str, k: int = 5, method: str = "tfidf") -> List[str]

    # --- Evaluasi Retrieval ---
    def eval_retrieval(queries, method="tfidf", k=5):
        eval_rows = []

        for i, q in enumerate(queries):
            query_text = q["query"]
            ground_truth = q["ground_truth"]
            retrieved = retrieve(query_text, k=k, method=method)
            is_correct = int(ground_truth in retrieved)

            eval_rows.append({
                "query_id": f"Q{i+1}",
                "query_text": query_text,
                "ground_truth": ground_truth,
                "top_k_case_ids": retrieved,
                "correct": is_correct
            })

        df_eval = pd.DataFrame(eval_rows)
        y_true = [1] * len(df_eval)
        y_pred = df_eval["correct"].tolist()

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }

        return df_eval, metrics

    # --- Jalankan Evaluasi Retrieval ---
    retrieval_eval_df, retrieval_metrics = eval_retrieval(eval_queries, method="tfidf", k=5)
    retrieval_csv = "/content/drive/MyDrive/UASPK/data/eval/retrieval_metrics.csv"
    retrieval_eval_df.to_csv(retrieval_csv, index=False)
    print(f"
retrieval_metrics.csv disimpan di: {retrieval_csv}")

    for m, v in retrieval_metrics.items():
        print(f"{m.capitalize()}: {v:.2%}")

    # --- Evaluasi Prediksi Solusi (Cosine Similarity) ---
    def eval_prediction(predictions_df):
        similarities = []

        for i, row in predictions_df.iterrows():
            actual = row["actual_solution"]
            predicted = row["predicted_solution"]

            tfidf = TfidfVectorizer().fit([actual, predicted])
            vecs = tfidf.transform([actual, predicted])
            score = cosine_similarity(vecs[0], vecs[1])[0][0]

            similarities.append(score)

        predictions_df["similarity_score"] = similarities
        return predictions_df

    # --- Load hasil prediksi ---
    pred_path = "/content/drive/MyDrive/UASPK/data/results/predictions.csv"
    predictions_df = pd.read_csv(pred_path)

    # --- Jika tersedia 'actual_solution', evaluasi
    if "actual_solution" in predictions_df.columns:
        eval_pred_df = eval_prediction(predictions_df)
        pred_eval_csv = "/content/drive/MyDrive/UASPK/data/eval/prediction_metrics.csv"
        eval_pred_df.to_csv(pred_eval_csv, index=False)
        print(f"
prediction_metrics.csv disimpan di: {pred_eval_csv}")
    else:
        print("
Kolom 'actual_solution' tidak ditemukan di predictions.csv — evaluasi prediksi dilewati.")

    # --- Bandingkan Model TF-IDF dan BERT ---
    def collect_metrics_for_models(models=["tfidf", "bert"], queries=None, k=5):
        summary = []
        for model in models:
            df_eval, metrics = eval_retrieval(queries, method=model, k=k)
            summary.append({
                "Model": model.upper(),
                "Accuracy": round(metrics["accuracy"], 3),
                "Precision": round(metrics["precision"], 3),
                "Recall": round(metrics["recall"], 3),
                "F1-score": round(metrics["f1_score"], 3)
            })
        return pd.DataFrame(summary)

    df_metrics = collect_metrics_for_models(models=["tfidf", "bert"], queries=eval_queries, k=5)
    print("
Perbandingan Model:
")
    print(df_metrics)

    # --- Visualisasi Bar Chart ---
    df_metrics.set_index("Model").plot(kind="bar", figsize=(8, 5),
        color=['#4caf50', '#2196f3', '#ff9800', '#f44336'])
    plt.title("Comparison of Retrieval Models")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # --- Analisis Error (Query yang Gagal) ---
    def show_errors(eval_df, queries, max_display=5):
        error_queries = eval_df[eval_df["correct"] == 0]
        display_count = min(len(error_queries), max_display)

        if display_count == 0:
            print("
Tidak ada retrieval yang gagal.")
            return

        print(f"
Menampilkan {display_count} dari {len(error_queries)} kasus error:
")
        for i in range(display_count):
            row = error_queries.iloc[i]
            query_id = row["query_id"]
            index = int(query_id[1:]) - 1
            query_text = queries[index]["query"]

            print(f"{query_id}:")
            print(f"   Query         : {query_text}")
            print(f"   Ground Truth  : {row['ground_truth']}")
            print(f"   Top-K Returned: {row['top_k_case_ids']}")
            print("-" * 70)

    show_errors(retrieval_eval_df, eval_queries)
