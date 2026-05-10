"""
src/fuse.py

Fuse UniXcoder embeddings (768-dim) with normalized AST features (8-dim)
into a single 776-dim fused representation.

Usage:
    python src/fuse.py --input data/processed/functions.jsonl \
                       --output data/processed/functions.jsonl \
                       --scaler data/processed/scaler.pkl
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

AST_FEATURE_ORDER = [
    "if_count",
    "loop_count",
    "return_count",
    "call_count",
    "ast_depth",
    "line_count",
    "param_count",
    "pointer_count",
]


def extract_ast_vector(fn: dict) -> list[float]:
    """Extract AST features in canonical order as a float list."""
    feats = fn.get("ast_features", {})
    return [float(feats.get(k, 0)) for k in AST_FEATURE_ORDER]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse embeddings + AST features into 776-dim vectors")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scaler", required=True, help="Path to save/load fitted StandardScaler (.pkl)")
    parser.add_argument("--load-scaler", action="store_true",
                        help="Load existing scaler instead of fitting a new one (for inference)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    scaler_path = Path(args.scaler)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading functions from %s", input_path)
    functions: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                functions.append(json.loads(line))
    log.info("Loaded %d functions", len(functions))

    # Validate embeddings present
    missing_emb = [fn.get("function_name") for fn in functions if "embedding" not in fn]
    if missing_emb:
        log.error("%d functions missing embeddings. Run embed.py first.", len(missing_emb))
        sys.exit(1)

    # Build AST feature matrix
    ast_matrix = np.array([extract_ast_vector(fn) for fn in functions], dtype=np.float32)
    log.info("AST feature matrix shape: %s", ast_matrix.shape)

    # Fit or load scaler
    if args.load_scaler and scaler_path.exists():
        log.info("Loading existing scaler from %s", scaler_path)
        with open(scaler_path, "rb") as f:
            scaler: StandardScaler = pickle.load(f)
        ast_normalized = scaler.transform(ast_matrix)
    else:
        log.info("Fitting StandardScaler on %d functions", len(functions))
        scaler = StandardScaler()
        ast_normalized = scaler.fit_transform(ast_matrix)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        log.info("Scaler saved to %s", scaler_path)

    # Build embedding matrix
    emb_matrix = np.array([fn["embedding"] for fn in functions], dtype=np.float32)
    log.info("Embedding matrix shape: %s", emb_matrix.shape)

    # Fuse: concatenate [768-dim embedding] + [8-dim normalized AST]
    fused_matrix = np.concatenate([emb_matrix, ast_normalized], axis=1)
    log.info("Fused matrix shape: %s  (expected 776-dim)", fused_matrix.shape)
    assert fused_matrix.shape[1] == 776, f"Expected 776-dim, got {fused_matrix.shape[1]}"

    # Write back
    for i, fn in enumerate(functions):
        functions[i]["fused_embedding"] = fused_matrix[i].tolist()

    log.info("Writing output to %s", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for fn in functions:
            f.write(json.dumps(fn) + "\n")

    log.info("Done. Fused embeddings written (%d functions, 776-dim).", len(functions))


if __name__ == "__main__":
    main()
