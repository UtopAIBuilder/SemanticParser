"""
extract.py — Deterministic function extraction from C source files.

Uses tree-sitter to parse C files and extract:
- Function name, code, file path
- AST features (control flow counts, depth, complexity metrics)

No ML models used here. Everything is deterministic from the C grammar.

Usage:
    python src/extract.py --raw-dir data/raw --output data/processed/functions.jsonl
"""

import argparse
import json
import os
from pathlib import Path

import tree_sitter_c as tsc
from tree_sitter import Language, Parser


# ---------------------------------------------------------------------------
# Parser setup
# ---------------------------------------------------------------------------

def get_parser() -> Parser:
    """Return configured tree-sitter C parser."""
    lang = Language(tsc.language())
    return Parser(lang)


# ---------------------------------------------------------------------------
# AST traversal utilities
# ---------------------------------------------------------------------------

def walk(node, visitor):
    """Recursively walk AST, calling visitor(node) on every node."""
    visitor(node)
    for child in node.children:
        walk(child, visitor)


def get_depth(node) -> int:
    """Return maximum depth of AST subtree rooted at node."""
    if not node.children:
        return 0
    return 1 + max(get_depth(child) for child in node.children)


# ---------------------------------------------------------------------------
# Function name extraction
# ---------------------------------------------------------------------------

def get_function_name(node, code_bytes: bytes) -> str:
    """
    Extract function name from a function_definition AST node.

    Handles three cases:
    1. Normal:   int foo(args)       → declarator=function_declarator
    2. Pointer:  int *foo(args)      → declarator=pointer_declarator → function_declarator
    3. Macro:    IRAM_ATTR foo(void) → type=type_identifier (name), declarator=parenthesized_declarator
                 Tree-sitter loses the return type to macro expansion and
                 misparses the function name as the type node.
    """
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return "<unknown>"

    # Case 3: macro attribute stripped return type — name is in type field
    if declarator.type == "parenthesized_declarator":
        type_node = node.child_by_field_name("type")
        if type_node is not None:
            return code_bytes[type_node.start_byte:type_node.end_byte].decode(
                "utf-8", errors="replace"
            ).strip()
        return "<unknown>"

    # Case 2: pointer return type — unwrap pointer_declarator chain
    while declarator.type == "pointer_declarator":
        declarator = declarator.child_by_field_name("declarator")
        if declarator is None:
            return "<unknown>"

    # Case 1: normal function_declarator
    if declarator.type == "function_declarator":
        inner = declarator.child_by_field_name("declarator")
        if inner is not None:
            return code_bytes[inner.start_byte:inner.end_byte].decode(
                "utf-8", errors="replace"
            )

    return "<unknown>"


# ---------------------------------------------------------------------------
# AST feature extraction
# ---------------------------------------------------------------------------

def compute_ast_features(func_node, code_bytes: bytes) -> dict:
    """
    Deterministically compute structural features from the function AST.

    Features:
        if_count      : number of if_statement nodes
        loop_count    : for + while + do_statement nodes
        return_count  : number of return_statement nodes
        call_count    : number of call_expression nodes
        ast_depth     : maximum depth of function AST subtree
        line_count    : source lines spanned by function
        param_count   : number of parameters
        pointer_count : number of pointer_declarator nodes (memory risk signal)
    """
    counts = {
        "if_count": 0,
        "loop_count": 0,
        "return_count": 0,
        "call_count": 0,
        "pointer_count": 0,
    }

    def count_nodes(node):
        t = node.type
        if t == "if_statement":
            counts["if_count"] += 1
        elif t in ("for_statement", "while_statement", "do_statement"):
            counts["loop_count"] += 1
        elif t == "return_statement":
            counts["return_count"] += 1
        elif t == "call_expression":
            counts["call_count"] += 1
        elif t == "pointer_declarator":
            counts["pointer_count"] += 1

    walk(func_node, count_nodes)

    # Parameter count from function_declarator → parameter_list
    param_count = 0
    declarator = func_node.child_by_field_name("declarator")
    while declarator and declarator.type == "pointer_declarator":
        declarator = declarator.child_by_field_name("declarator")
    if declarator and declarator.type == "function_declarator":
        param_list = declarator.child_by_field_name("parameters")
        if param_list:
            # named children excludes punctuation like commas and parens
            param_count = sum(
                1 for c in param_list.named_children
                if c.type == "parameter_declaration"
            )

    # Line count
    line_count = func_node.end_point[0] - func_node.start_point[0] + 1

    # AST depth (can be slow on very large functions — acceptable for 26 files)
    ast_depth = get_depth(func_node)

    return {
        "if_count":      counts["if_count"],
        "loop_count":    counts["loop_count"],
        "return_count":  counts["return_count"],
        "call_count":    counts["call_count"],
        "ast_depth":     ast_depth,
        "line_count":    line_count,
        "param_count":   param_count,
        "pointer_count": counts["pointer_count"],
    }


# ---------------------------------------------------------------------------
# Control flow elements (for labels.control_flow_elements)
# ---------------------------------------------------------------------------

def compute_control_flow_elements(func_node) -> list:
    """
    Return sorted list of control flow keywords present in function.
    Deterministic — used directly as a label, no model needed.
    """
    elements = set()

    def find_cf(node):
        t = node.type
        if t == "if_statement":
            elements.add("if")
        elif t == "for_statement":
            elements.add("for")
        elif t == "while_statement":
            elements.add("while")
        elif t == "do_statement":
            elements.add("do_while")
        elif t == "switch_statement":
            elements.add("switch")
        elif t == "return_statement":
            elements.add("return")
        elif t == "break_statement":
            elements.add("break")
        elif t == "continue_statement":
            elements.add("continue")
        elif t == "goto_statement":
            elements.add("goto")

    walk(func_node, find_cf)
    return sorted(elements)


# ---------------------------------------------------------------------------
# Call expression extraction (used by label.py for side_effect detection)
# ---------------------------------------------------------------------------

def extract_call_names(func_node, code_bytes: bytes) -> list:
    """
    Extract all called function names from call_expression nodes.

    Handles:
        direct calls   : foo(args)         → "foo"
        field calls    : obj->method(args) → "obj->method"
        pointer calls  : (*fn)(args)       → kept as raw text, short form

    Returns list of callee name strings (may have duplicates — caller dedupes).
    """
    call_names = []

    def find_calls(node):
        if node.type == "call_expression":
            fn_node = node.child_by_field_name("function")
            if fn_node is not None:
                name = code_bytes[fn_node.start_byte:fn_node.end_byte].decode(
                    "utf-8", errors="replace"
                ).strip()
                call_names.append(name)

    walk(func_node, find_calls)
    return call_names


# ---------------------------------------------------------------------------
# Per-file extraction
# ---------------------------------------------------------------------------

def extract_functions(file_path: str, parser: Parser) -> list:
    """
    Parse a single .c file and return list of function datapoints.

    Each datapoint:
        function_name         : str
        function_code         : str
        file_path             : str
        ast_features          : dict (8 numeric features)
        control_flow_elements : list[str] (deterministic label)
        call_names            : list[str] (for side_effect detection in label.py)
    """
    with open(file_path, "rb") as f:
        code_bytes = f.read()

    tree = parser.parse(code_bytes)
    root = tree.root_node

    functions = []

    def visit(node):
        if node.type == "function_definition":
            name = get_function_name(node, code_bytes)
            code = code_bytes[node.start_byte:node.end_byte].decode(
                "utf-8", errors="replace"
            )
            ast_features = compute_ast_features(node, code_bytes)
            cf_elements = compute_control_flow_elements(node)
            call_names = extract_call_names(node, code_bytes)

            functions.append({
                "function_name":         name,
                "function_code":         code,
                "file_path":             str(file_path),
                "ast_features":          ast_features,
                "control_flow_elements": cf_elements,
                "call_names":            call_names,
            })

    walk(root, visit)
    return functions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_extraction(raw_dir: str, output_path: str):
    parser = get_parser()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    all_functions = []
    c_files = sorted(Path(raw_dir).glob("*.c"))

    if not c_files:
        print(f"No .c files found in {raw_dir}")
        return

    print(f"Found {len(c_files)} .c files")

    for c_file in c_files:
        funcs = extract_functions(str(c_file), parser)
        print(f"  {c_file.name:<40} → {len(funcs):>4} functions")
        all_functions.extend(funcs)

    print(f"\nTotal functions extracted: {len(all_functions)}")

    # Write one JSON object per line
    with open(output_path, "w") as f:
        for fn in all_functions:
            f.write(json.dumps(fn) + "\n")

    print(f"Written to {output_path}")

    # Quick sanity stats
    total_lines = sum(fn["ast_features"]["line_count"] for fn in all_functions)
    avg_lines   = total_lines / len(all_functions) if all_functions else 0
    max_lines   = max(fn["ast_features"]["line_count"] for fn in all_functions)
    print(f"\nDataset stats:")
    print(f"  Avg function length : {avg_lines:.1f} lines")
    print(f"  Max function length : {max_lines} lines")
    print(f"  Functions with loops: {sum(1 for f in all_functions if f['ast_features']['loop_count'] > 0)}")
    print(f"  Functions with ptrs : {sum(1 for f in all_functions if f['ast_features']['pointer_count'] > 0)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract C functions using tree-sitter")
    ap.add_argument("--raw-dir", default="data/raw",       help="Directory of .c files")
    ap.add_argument("--output",  default="data/processed/functions.jsonl", help="Output JSONL path")
    args = ap.parse_args()
    run_extraction(args.raw_dir, args.output)
