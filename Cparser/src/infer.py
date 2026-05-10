"""
src/infer.py

CLI inference interface. Given a .c file, runs the full pipeline:
  extract → label → embed → fuse → classify
and prints structured JSON to stdout.

Usage:
    python src/infer.py --file sample.c
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Reuse shared logic from other modules
sys.path.insert(0, str(Path(__file__).parent))

from label import (
    extract_call_names_from_code,
    deterministic_side_effects,
    llm_side_effects_fallback,
    llm_high_level_purpose,
)

logging.basicConfig(
    level=logging.WARNING,   # quiet for CLI — errors only
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_NAME = "microsoft/unixcoder-base"
MAX_LENGTH = 512
AST_FEATURE_ORDER = [
    "if_count", "loop_count", "return_count", "call_count",
    "ast_depth", "line_count", "param_count", "pointer_count",
]


# ---------------------------------------------------------------------------
# Extract functions from a .c file using tree-sitter
# ---------------------------------------------------------------------------

def extract_functions_from_file(c_path: Path) -> list[dict]:
    """Extract functions from a .c file. Returns list of dicts matching functions.jsonl schema."""
    from tree_sitter import Language, Parser
    import tree_sitter_c as tsc

    lang = Language(tsc.language())
    parser = Parser(lang)

    code = c_path.read_text(encoding="utf-8", errors="replace")
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)

    functions = []

    def walk(node):
        if node.type == "function_definition":
            fn = _parse_function_node(node, code_bytes, str(c_path))
            if fn:
                functions.append(fn)
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return functions


def _parse_function_node(node, code_bytes: bytes, file_path: str) -> dict | None:
    """Parse a single function_definition node into a dict."""
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return None

    # Resolve function name (same three cases as extract.py)
    name = _get_function_name(node, declarator, code_bytes)
    if name is None:
        return None

    code = code_bytes[node.start_byte: node.end_byte].decode("utf-8", errors="replace")
    ast_features = _compute_ast_features(node, code_bytes)
    control_flow = _get_control_flow_elements(node)

    return {
        "function_name": name,
        "function_code": code,
        "file_path": file_path,
        "ast_features": ast_features,
        "control_flow_elements": control_flow,
    }


def _get_function_name(fn_node, declarator, code_bytes: bytes) -> str | None:
    if declarator.type == "function_declarator":
        id_node = declarator.child_by_field_name("declarator")
        if id_node and id_node.type == "identifier":
            return code_bytes[id_node.start_byte: id_node.end_byte].decode()
    elif declarator.type == "pointer_declarator":
        inner = declarator.child_by_field_name("declarator")
        if inner and inner.type == "function_declarator":
            id_node = inner.child_by_field_name("declarator")
            if id_node and id_node.type == "identifier":
                return code_bytes[id_node.start_byte: id_node.end_byte].decode()
    elif declarator.type == "parenthesized_declarator":
        type_node = fn_node.child_by_field_name("type")
        if type_node:
            return code_bytes[type_node.start_byte: type_node.end_byte].decode()
    return None


def _count_nodes(node, node_type: str) -> int:
    count = 1 if node.type == node_type else 0
    for child in node.children:
        count += _count_nodes(child, node_type)
    return count


def _ast_depth(node, depth: int = 0) -> int:
    if not node.children:
        return depth
    return max(_ast_depth(child, depth + 1) for child in node.children)


def _get_params(node) -> int:
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return 0
    fd = declarator if declarator.type == "function_declarator" else None
    if fd is None and declarator.type == "pointer_declarator":
        fd = declarator.child_by_field_name("declarator")
    if fd is None:
        return 0
    params_node = fd.child_by_field_name("parameters")
    if params_node is None:
        return 0
    return sum(1 for c in params_node.children if c.type == "parameter_declaration")


def _count_pointers(node) -> int:
    count = 1 if node.type == "pointer_declarator" else 0
    for child in node.children:
        count += _count_pointers(child)
    return count


def _compute_ast_features(node, code_bytes: bytes) -> dict:
    code = code_bytes[node.start_byte: node.end_byte].decode("utf-8", errors="replace")
    return {
        "if_count": _count_nodes(node, "if_statement"),
        "loop_count": (
            _count_nodes(node, "for_statement")
            + _count_nodes(node, "while_statement")
            + _count_nodes(node, "do_statement")
        ),
        "return_count": _count_nodes(node, "return_statement"),
        "call_count": _count_nodes(node, "call_expression"),
        "ast_depth": _ast_depth(node),
        "line_count": code.count("\n") + 1,
        "param_count": _get_params(node),
        "pointer_count": _count_pointers(node),
    }


def _get_control_flow_elements(node) -> list[str]:
    elements = set()
    for ntype, label in [
        ("if_statement", "if"),
        ("for_statement", "for"),
        ("while_statement", "while"),
        ("do_statement", "do_while"),
        ("switch_statement", "switch"),
        ("return_statement", "return"),
        ("break_statement", "break"),
        ("continue_statement", "continue"),
        ("goto_statement", "goto"),
    ]:
        if _count_nodes(node, ntype) > 0:
            elements.add(label)
    return sorted(elements)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None
_device = None


def _load_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    _device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModel.from_pretrained(MODEL_NAME)
    _model.to(_device)
    _model.eval()


def embed_code(code: str) -> list[float]:
    _load_model()
    inputs = _tokenizer(
        code,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(_device)
    with torch.no_grad():
        outputs = _model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().squeeze(0).tolist()


# ---------------------------------------------------------------------------
# Fuse
# ---------------------------------------------------------------------------

def fuse_embedding(embedding: list[float], ast_features: dict, scaler) -> list[float]:
    ast_vec = np.array([[float(ast_features.get(k, 0)) for k in AST_FEATURE_ORDER]], dtype=np.float32)
    ast_norm = scaler.transform(ast_vec)[0].tolist()
    return embedding + ast_norm


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------

def classify(fused_embedding: list[float], classifier_payload: dict) -> list[str]:
    clf = classifier_payload["classifier"]
    mlb = classifier_payload["mlb"]
    X = np.array([fused_embedding], dtype=np.float32)
    Y_pred = clf.predict(X)
    labels = mlb.inverse_transform(Y_pred)
    return list(labels[0]) if labels[0] else ["none"]


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------

def load_artifacts():
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    clf_path = ARTIFACTS_DIR / "classifier.pkl"

    if not scaler_path.exists():
        print(f"ERROR: scaler not found at {scaler_path}. Run fuse.py first.", file=sys.stderr)
        sys.exit(1)
    if not clf_path.exists():
        print(f"ERROR: classifier not found at {clf_path}. Run eval.py first.", file=sys.stderr)
        sys.exit(1)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(clf_path, "rb") as f:
        clf_payload = pickle.load(f)

    return scaler, clf_payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Infer function properties from a .c file")
    parser.add_argument("--file", required=True, help="Path to .c file")
    args = parser.parse_args()

    c_path = Path(args.file)
    if not c_path.exists():
        print(f"ERROR: file not found: {c_path}", file=sys.stderr)
        sys.exit(1)

    scaler, clf_payload = load_artifacts()

    # Step 1: Extract
    functions = extract_functions_from_file(c_path)
    if not functions:
        print(json.dumps({"functions": [], "message": "No functions found"}))
        return

    output_functions = []

    for fn in functions:
        code = fn["function_code"]

        # Step 2: Label
        call_names = extract_call_names_from_code(code)
        side_effects, is_ambiguous = deterministic_side_effects(call_names)
        if is_ambiguous:
            side_effects = llm_side_effects_fallback(code)
        high_level_purpose = llm_high_level_purpose(code)

        # Step 3: Embed
        embedding = embed_code(code)

        # Step 4: Fuse
        fused = fuse_embedding(embedding, fn["ast_features"], scaler)

        # Step 5: Classify
        predicted_side_effects = classify(fused, clf_payload)

        output_functions.append({
            "name": fn["function_name"],
            "high_level_purpose": high_level_purpose,
            "predicted_side_effects": predicted_side_effects,
            "control_flow_elements": fn["control_flow_elements"],
            "ast_features": fn["ast_features"],
            "embedding_dim": len(fused),
            "embedding_preview": embedding[:3],
        })

    print(json.dumps({"functions": output_functions}, indent=2))


if __name__ == "__main__":
    main()
