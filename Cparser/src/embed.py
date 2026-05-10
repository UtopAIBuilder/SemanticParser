"""
src/embed.py

Generate 768-dim UniXcoder embeddings (CLS token) for each function.
Uses MPS on Apple Silicon, falls back to CPU.

Usage:
    python src/embed.py --input data/processed/functions.jsonl \
                        --output data/processed/functions.jsonl
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

MODEL_NAME = "microsoft/unixcoder-base"
MAX_LENGTH = 512
BATCH_SIZE = 8  # tune down if OOM on MPS


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        log.info("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    log.info("MPS not available, using CPU")
    return torch.device("cpu")


def load_model(device: torch.device):
    log.info("Loading UniXcoder tokenizer and model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    log.info("Model loaded")
    return tokenizer, model


def embed_batch(
    codes: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> list[list[float]]:
    """Embed a batch of code strings. Returns list of 768-dim float lists."""
    inputs = tokenizer(
        codes,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token at position 0
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
    return cls_embeddings.cpu().tolist()


def check_truncation(code: str, tokenizer) -> bool:
    """Return True if code would be truncated at 512 tokens."""
    ids = tokenizer.encode(code, add_special_tokens=True)
    return len(ids) > MAX_LENGTH


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UniXcoder embeddings for functions")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume", action="store_true", help="Skip functions that already have embeddings")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading functions from %s", input_path)
    functions: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                functions.append(json.loads(line))
    log.info("Loaded %d functions", len(functions))

    device = get_device()
    tokenizer, model = load_model(device)

    # Identify functions to embed
    to_embed_indices = []
    for i, fn in enumerate(functions):
        if args.resume and "embedding" in fn:
            continue
        to_embed_indices.append(i)

    log.info("%d functions to embed", len(to_embed_indices))

    truncated_count = 0
    t_start = time.time()

    for batch_start in range(0, len(to_embed_indices), args.batch_size):
        batch_indices = to_embed_indices[batch_start: batch_start + args.batch_size]
        codes = [functions[i]["function_code"] for i in batch_indices]

        # Check for truncation in this batch
        for i, code in zip(batch_indices, codes):
            if check_truncation(code, tokenizer):
                truncated_count += 1
                log.warning(
                    "Truncation (>512 tokens): %s in %s",
                    functions[i].get("function_name"),
                    functions[i].get("file_path"),
                )
                functions[i]["truncated"] = True

        embeddings = embed_batch(codes, tokenizer, model, device)

        for i, emb in zip(batch_indices, embeddings):
            functions[i]["embedding"] = emb

        processed = batch_start + len(batch_indices)
        log.info(
            "[%d/%d] embedded batch  (%.1fs elapsed)",
            processed,
            len(to_embed_indices),
            time.time() - t_start,
        )

    log.info("Embedding complete. %d functions truncated at 512 tokens.", truncated_count)

    log.info("Writing output to %s", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for fn in functions:
            f.write(json.dumps(fn) + "\n")

    log.info("Done. Total time: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
