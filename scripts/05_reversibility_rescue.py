#!/usr/bin/env python
"""Optional task-relevance retention analysis using downstream neighborhoods."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", default="outputs/processed/edges.parquet")
    parser.add_argument("--out-dir", default="outputs/reversibility_rescue")
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    edges = pd.read_parquet(args.edges, columns=["src", "dst", "weight", "src_type", "dst_type"]).dropna()
    edges = edges[(edges["src_type"].astype(str).str.len() > 0) & (edges["dst_type"].astype(str).str.len() > 0)]
    if len(edges) > args.max_rows:
        edges = edges.sample(args.max_rows, random_state=args.seed)
    if edges["src_type"].nunique() < 2 or edges["dst_type"].nunique() < 2:
        (out / "rescue_summary.csv").write_text("status,message\nskipped,insufficient typed labels\n", encoding="utf-8")
        print("Skipped rescue analysis: insufficient typed labels")
        return
    top_dst = edges["dst_type"].value_counts().head(100).index
    use = edges[edges["dst_type"].isin(top_dst)].copy()
    X = pd.crosstab(use["src"], use["dst_type"], values=use["weight"], aggfunc="sum").fillna(0)
    y = use.drop_duplicates("src").set_index("src")["src_type"].reindex(X.index).astype(str)
    keep_labels = y.value_counts()[lambda s: s >= 5].index
    X = X[y.isin(keep_labels)]
    y = y[y.isin(keep_labels)]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=args.seed, stratify=y_enc)
    models = {
        "real_downstream_neighborhood": LogisticRegression(max_iter=500, class_weight="balanced"),
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro"),
                "cross_entropy": log_loss(y_test, proba, labels=model.classes_) if proba is not None else float("nan"),
                "n_samples": len(X),
                "n_classes": len(le.classes_),
            }
        )
    pd.DataFrame(rows).to_csv(out / "rescue_summary.csv", index=False)
    print(f"Wrote optional rescue analysis to {out}")


if __name__ == "__main__":
    main()
