"""
src/eval.py

Three-axis evaluation of the fused embedding model:
  1. Classification  — multilabel side_effects prediction (LogisticRegression, one-vs-rest)
  2. Retrieval       — MRR@5, Recall@3, Recall@5 on fused embeddings
  3. Clustering      — K-means + UMAP visual sanity check

Also saves artifacts needed by infer.py:
  - data/processed/classifier.pkl
  - data/processed/scaler.pkl          (already saved by fuse.py)
  - data/processed/all_embeddings.npy
  - data/processed/all_labels.json

Usage:
    python src/eval.py --input data/processed/functions.jsonl \
                       --output-dir results/
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

AST_FEATURE_ORDER = [
    "if_count", "loop_count", "return_count", "call_count",
    "ast_depth", "line_count", "param_count", "pointer_count",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(jsonl_path: Path):
    functions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                functions.append(json.loads(line))
    return functions


def build_matrices(functions: list[dict]):
    """Return fused_emb, ast_emb, code_emb, label_lists, function_names."""
    fused, ast_vecs, code_embs = [], [], []
    label_lists = []

    for fn in functions:
        fused.append(fn["fused_embedding"])
        code_embs.append(fn["embedding"])
        feats = fn.get("ast_features", {})
        ast_vecs.append([float(feats.get(k, 0)) for k in AST_FEATURE_ORDER])
        side_effects = fn.get("labels", {}).get("side_effects", ["none"])
        label_lists.append(side_effects)

    return (
        np.array(fused, dtype=np.float32),
        np.array(ast_vecs, dtype=np.float32),
        np.array(code_embs, dtype=np.float32),
        label_lists,
        [fn.get("function_name", "?") for fn in functions],
    )


# ---------------------------------------------------------------------------
# Axis 1: Classification
# ---------------------------------------------------------------------------

def eval_classification(
    fused_emb: np.ndarray,
    ast_emb: np.ndarray,
    code_emb: np.ndarray,
    label_lists: list,
    output_dir: Path,
) -> dict:
    """Multilabel classification ablation: AST-only, code-only, fused."""
    log.info("=== Axis 1: Classification ===")

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_lists)
    classes = list(mlb.classes_)
    log.info("Classes: %s", classes)
    log.info("Label distribution:\n%s", {c: int(Y[:, i].sum()) for i, c in enumerate(classes)})

    # Normalize AST for AST-only baseline
    scaler_ast = StandardScaler()
    ast_norm = scaler_ast.fit_transform(ast_emb)

    results = {}

    variants = [
        ("AST-only (8-dim)", ast_norm),
        ("UniXcoder-only (768-dim)", code_emb),
        ("Fused (776-dim)", fused_emb),
    ]

    # Train/test split (stratified is hard with multilabel, use random)
    split_idx = int(0.8 * len(fused_emb))
    idx = np.arange(len(fused_emb))
    np.random.seed(42)
    np.random.shuffle(idx)
    train_idx, test_idx = idx[:split_idx], idx[split_idx:]

    Y_train, Y_test = Y[train_idx], Y[test_idx]

    trained_classifier = None

    for name, X in variants:
        X_train, X_test = X[train_idx], X[test_idx]
        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        )
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        report = classification_report(
            Y_test, Y_pred, target_names=classes, output_dict=True, zero_division=0
        )
        macro_f1 = report["macro avg"]["f1-score"]
        micro_f1 = report.get("weighted avg", {}).get("f1-score", 0.0)

        log.info("\n--- %s ---", name)
        log.info(classification_report(Y_test, Y_pred, target_names=classes, zero_division=0))

        results[name] = {
            "macro_f1": round(macro_f1, 4),
            "micro_f1": round(micro_f1, 4),
            "per_class": {c: round(report[c]["f1-score"], 4) for c in classes if c in report},
        }

        if name == "Fused (776-dim)":
            trained_classifier = clf

    # Save the fused classifier
    clf_path = output_dir.parent / "data" / "processed" / "classifier.pkl"
    clf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(clf_path, "wb") as f:
        pickle.dump({"classifier": trained_classifier, "mlb": mlb, "classes": classes}, f)
    log.info("Classifier saved to %s", clf_path)

    # Save ablation table
    table_path = output_dir / "classification_ablation.json"
    with open(table_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Ablation table saved to %s", table_path)

    # Failure analysis: lowest F1 class
    fused_result = results.get("Fused (776-dim)", {})
    per_class = fused_result.get("per_class", {})
    if per_class:
        worst_class = min(per_class, key=lambda c: per_class[c])
        log.info(
            "Failure analysis: lowest F1 class = '%s' (F1=%.4f). "
            "Likely cause: fewer training examples or overlap with another class.",
            worst_class, per_class[worst_class],
        )

    return results


# ---------------------------------------------------------------------------
# Axis 2: Retrieval
# ---------------------------------------------------------------------------

def mean_reciprocal_rank(
    query_idx: int,
    embeddings: np.ndarray,
    labels: list[list[str]],
    k: int = 5,
) -> float:
    """
    1. Cosine similarity to all others
    2. Rank descending, exclude self
    3. Find rank of first relevant (same dominant label)
    4. Return 1/rank (0 if not found in top-k)
    """
    sims = cosine_similarity(embeddings[query_idx: query_idx + 1], embeddings)[0]
    sims[query_idx] = -1.0  # exclude self

    ranked = np.argsort(sims)[::-1][:k]
    query_label = set(labels[query_idx])

    for rank, idx in enumerate(ranked, start=1):
        if set(labels[idx]) & query_label:  # any shared label
            return 1.0 / rank
    return 0.0


def recall_at_k(
    query_idx: int,
    embeddings: np.ndarray,
    labels: list[list[str]],
    k: int,
) -> float:
    """Recall@K: fraction of relevant docs in top-K (capped at K)."""
    sims = cosine_similarity(embeddings[query_idx: query_idx + 1], embeddings)[0]
    sims[query_idx] = -1.0

    ranked = np.argsort(sims)[::-1][:k]
    query_label = set(labels[query_idx])

    # All relevant (sharing at least one label), excluding self
    all_relevant = [
        i for i in range(len(labels))
        if i != query_idx and set(labels[i]) & query_label
    ]
    if not all_relevant:
        return 0.0

    hits = sum(1 for idx in ranked if set(labels[idx]) & query_label)
    return hits / min(len(all_relevant), k)


def eval_retrieval(
    fused_emb: np.ndarray,
    label_lists: list,
    function_names: list[str],
    output_dir: Path,
) -> dict:
    log.info("=== Axis 2: Retrieval ===")

    mrr_scores = []
    recall3_scores = []
    recall5_scores = []
    worst_mrr = 1.0
    worst_query_idx = 0

    for i in range(len(fused_emb)):
        mrr = mean_reciprocal_rank(i, fused_emb, label_lists, k=5)
        r3 = recall_at_k(i, fused_emb, label_lists, k=3)
        r5 = recall_at_k(i, fused_emb, label_lists, k=5)
        mrr_scores.append(mrr)
        recall3_scores.append(r3)
        recall5_scores.append(r5)
        if mrr < worst_mrr:
            worst_mrr = mrr
            worst_query_idx = i

    metrics = {
        "MRR@5": round(float(np.mean(mrr_scores)), 4),
        "Recall@3": round(float(np.mean(recall3_scores)), 4),
        "Recall@5": round(float(np.mean(recall5_scores)), 4),
    }
    log.info("Retrieval metrics: %s", metrics)

    # Failure analysis: show worst query
    sims = cosine_similarity(fused_emb[worst_query_idx: worst_query_idx + 1], fused_emb)[0]
    sims[worst_query_idx] = -1.0
    top5 = np.argsort(sims)[::-1][:5]
    log.info(
        "Failure analysis — worst retrieval query: '%s' (MRR=%.4f)\n"
        "  query labels: %s\n"
        "  top-5 retrieved: %s",
        function_names[worst_query_idx],
        worst_mrr,
        label_lists[worst_query_idx],
        [(function_names[j], label_lists[j]) for j in top5],
    )

    results_path = output_dir / "retrieval_metrics.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Retrieval results saved to %s", results_path)

    return metrics


# ---------------------------------------------------------------------------
# Axis 3: Clustering
# ---------------------------------------------------------------------------

def plot_umap(
    embeddings: np.ndarray,
    labels: list[list[str]],
    title: str,
    output_path: Path,
) -> None:
    try:
        import umap
    except ImportError:
        log.warning("umap-learn not installed. Skipping UMAP plot.")
        return

    log.info("Running UMAP dimensionality reduction…")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    # Get dominant label per function
    all_labels_flat = sorted({lbl for ls in labels for lbl in ls})
    color_map = {lbl: i for i, lbl in enumerate(all_labels_flat)}
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_labels_flat)))

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl_name in all_labels_flat:
        mask = [any(l == lbl_name for l in ls) for ls in labels]
        pts = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], label=lbl_name,
                   color=colors[color_map[lbl_name]], alpha=0.6, s=20)

    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info("UMAP plot saved to %s", output_path)


def eval_clustering(
    fused_emb: np.ndarray,
    label_lists: list,
    function_names: list[str],
    output_dir: Path,
) -> dict:
    log.info("=== Axis 3: Clustering ===")

    k = 3  # io, memory, hardware
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(fused_emb)

    # Map each function's dominant side_effect label to an int
    priority = {"hardware": 0, "io": 1, "memory": 2, "none": 3}
    dominant = [
        min(ls, key=lambda x: priority.get(x, 99))
        for ls in label_lists
    ]

    # Measure cluster purity
    from collections import Counter

    cluster_info = {}
    for c in range(k):
        mask = cluster_labels == c
        contents = [dominant[i] for i in range(len(dominant)) if mask[i]]
        most_common = Counter(contents).most_common(1)[0] if contents else ("?", 0)
        purity = most_common[1] / len(contents) if contents else 0
        cluster_info[f"cluster_{c}"] = {
            "size": int(mask.sum()),
            "dominant_label": most_common[0],
            "purity": round(purity, 3),
            "distribution": dict(Counter(contents)),
        }
        log.info("Cluster %d: %s", c, cluster_info[f"cluster_{c}"])

    # Failure analysis: find a function in the wrong cluster
    for i, (fn_name, dom, cl) in enumerate(zip(function_names, dominant, cluster_labels)):
        # wrong cluster = dominant label differs from cluster's dominant label
        cluster_dom = cluster_info[f"cluster_{cl}"]["dominant_label"]
        if dom != cluster_dom:
            log.info(
                "Failure analysis — misclassified function: '%s' (true=%s, cluster=%d dominant=%s). "
                "Likely cause: function has mixed signals or call pattern overlaps multiple domains.",
                fn_name, dom, cl, cluster_dom,
            )
            break

    # UMAP plot
    plot_umap(fused_emb, label_lists, "Fused Embeddings — UMAP (colored by side_effect)",
              output_dir / "umap_fused.png")

    results_path = output_dir / "clustering_info.json"
    with open(results_path, "w") as f:
        json.dump(cluster_info, f, indent=2)
    log.info("Clustering info saved to %s", results_path)

    return cluster_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate embeddings: classification, retrieval, clustering")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    functions = load_data(input_path)
    log.info("Loaded %d functions", len(functions))

    # Validate required fields
    for field in ("embedding", "fused_embedding"):
        missing = [fn.get("function_name") for fn in functions if field not in fn]
        if missing:
            log.error("%d functions missing '%s'. Run upstream steps first.", len(missing), field)
            sys.exit(1)

    fused_emb, ast_emb, code_emb, label_lists, fn_names = build_matrices(functions)

    # Save artifacts for infer.py
    processed_dir = input_path.parent
    np.save(str(processed_dir / "all_embeddings.npy"), fused_emb)
    with open(processed_dir / "all_labels.json", "w") as f:
        json.dump(label_lists, f)
    log.info("Saved all_embeddings.npy and all_labels.json to %s", processed_dir)

    # Axis 1
    clf_results = eval_classification(fused_emb, ast_emb, code_emb, label_lists, output_dir)

    # Axis 2
    ret_results = eval_retrieval(fused_emb, label_lists, fn_names, output_dir)

    # Axis 3
    cluster_results = eval_clustering(fused_emb, label_lists, fn_names, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print("\n--- Classification (Fused) ---")
    fused_clf = clf_results.get("Fused (776-dim)", {})
    print(f"  Macro F1 : {fused_clf.get('macro_f1')}")
    print(f"  Micro F1 : {fused_clf.get('micro_f1')}")
    print("\n--- Retrieval ---")
    for k, v in ret_results.items():
        print(f"  {k}: {v}")
    print("\n--- Ablation ---")
    for variant, res in clf_results.items():
        print(f"  {variant}: macro_F1={res['macro_f1']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
