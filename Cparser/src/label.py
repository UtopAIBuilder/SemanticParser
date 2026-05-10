"""
src/label.py

Label functions with:
  - side_effects: deterministic from call_expression AST nodes, LLM fallback if ambiguous
  - high_level_purpose: qwen2.5-coder:7b with Chain-of-Thought, always
  - control_flow_elements: already in extract.py output, preserved here

Usage:
    python src/label.py --input data/processed/functions.jsonl \
                        --output data/processed/functions.jsonl

    # Restart from checkpoint after a crash:
    python src/label.py --input data/processed/functions.jsonl \
                        --output data/processed/functions.jsonl --resume
"""

import argparse
import json
import logging
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

CHECKPOINT_EVERY = 50

# ---------------------------------------------------------------------------
# Signal lists for deterministic side-effect detection
# ---------------------------------------------------------------------------

MEMORY_SIGNALS = {
    "malloc", "free", "realloc", "calloc",
    "heap_caps_malloc", "heap_caps_free", "heap_caps_calloc", "heap_caps_realloc",
    "pvPortMalloc", "vPortFree",
}

IO_SIGNALS = {
    "printf", "fprintf", "fwrite", "fread", "write", "read",
    "fopen", "fclose", "puts", "fputs",
    "ESP_LOGI", "ESP_LOGE", "ESP_LOGW", "ESP_LOGD", "ESP_LOGV",
    "esp_log_write", "uart_write_bytes", "uart_read_bytes",
}

HW_SIGNALS_EXACT = {
    "gpio_set_level", "gpio_get_level", "gpio_config",
    "gpio_set_direction", "gpio_set_pull_mode",
    "uart_param_config", "uart_driver_install", "uart_driver_delete",
    "spi_bus_initialize", "spi_bus_free", "spi_device_transmit",
    "i2c_master_cmd_begin", "i2c_master_start", "i2c_master_stop",
    "REG_WRITE", "REG_READ", "WRITE_PERI_REG", "READ_PERI_REG",
    "SET_PERI_REG_MASK", "CLEAR_PERI_REG_MASK",
    "esp_rom_gpio_pad_select_gpio",
    "ledc_set_duty", "ledc_update_duty",
    "adc1_get_raw", "adc2_get_raw",
    "rmt_write_items", "rmt_rx_start", "rmt_tx_start",
    "timer_set_counter_value", "timer_start", "timer_pause",
    "esp_intr_alloc", "esp_intr_free", "esp_intr_enable", "esp_intr_disable",
}

HW_PREFIXES = {
    "HAL_", "gpio_", "uart_", "spi_", "i2c_", "ledc_", "adc_",
    "rmt_", "timer_", "esp_rom_gpio",
}


# ---------------------------------------------------------------------------
# Deterministic side-effect labeling
# ---------------------------------------------------------------------------

def deterministic_side_effects(call_names: list[str]) -> tuple[list[str], bool]:
    """
    Match call_names against signal lists.

    Returns:
        matched      - list of matched side_effect labels (may have multiple)
        is_ambiguous - True when no signals matched but calls exist
    """
    matched: set[str] = set()

    for name in call_names:
        if name in MEMORY_SIGNALS:
            matched.add("memory")
        if name in IO_SIGNALS:
            matched.add("io")
        if name in HW_SIGNALS_EXACT:
            matched.add("hardware")
        for prefix in HW_PREFIXES:
            if name.startswith(prefix):
                matched.add("hardware")
                break

    is_ambiguous = (len(matched) == 0) and (len(call_names) > 0)
    return sorted(matched), is_ambiguous


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

OLLAMA_BASE = "http://localhost:11434"
LABEL_MODEL = "qwen2.5-coder:7b"


def ollama_generate(prompt: str, model: str = LABEL_MODEL) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def ollama_chat(messages: list[dict], model: str = LABEL_MODEL) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


# ---------------------------------------------------------------------------
# LLM-based labeling
# ---------------------------------------------------------------------------

_SIDE_EFFECT_CHOICES = {"io", "memory", "hardware", "none"}


def llm_side_effects_fallback(function_code: str) -> list[str]:
    """
    Called ONLY when deterministic labeling is ambiguous.
    Uses qwen2.5-coder:7b via Ollama /api/generate.
    """
    prompt = (
        "You are an expert in embedded C. Analyze this function and identify "
        "its side effects. A side effect is an observable interaction beyond "
        "the return value:\n"
        "  - io: file/serial/log operations\n"
        "  - memory: heap allocation or deallocation\n"
        "  - hardware: GPIO, UART, SPI, register access, interrupts\n"
        "  - none: pure computation, no external interaction\n\n"
        f"Function:\n{function_code}\n\n"
        'Respond ONLY with valid JSON, no other text:\n'
        '{"side_effects": ["io"|"memory"|"hardware"|"none"]}'
    )
    raw = ollama_generate(prompt)
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        effects = parsed.get("side_effects", [])
        validated = [e for e in effects if e in _SIDE_EFFECT_CHOICES]
        return validated if validated else ["none"]
    except (json.JSONDecodeError, KeyError):
        log.warning("LLM side_effects parse failed, raw: %r — defaulting to ['none']", raw[:200])
        return ["none"]


def llm_high_level_purpose(function_code: str) -> str:
    """
    Two-turn CoT prompt to qwen2.5-coder:7b via /api/chat.
    Turn 1: reasoning; Turn 2: extract one-sentence summary.
    """
    turn1_prompt = (
        "You are analyzing embedded C code. Given this function:\n"
        f"<code>\n{function_code}\n</code>\n\n"
        "Think step by step:\n"
        "1. What is the primary responsibility of this function?\n"
        "2. What does it set up, compute, or control?\n"
        "3. Who would call this function and why?\n\n"
        "Write your reasoning."
    )
    messages = [{"role": "user", "content": turn1_prompt}]
    turn1_response = ollama_chat(messages)

    turn2_prompt = (
        "Based on your reasoning, write ONE concise sentence describing what "
        "this function does. Start with a verb. No JSON, just the sentence."
    )
    messages = [
        {"role": "user", "content": turn1_prompt},
        {"role": "assistant", "content": turn1_response},
        {"role": "user", "content": turn2_prompt},
    ]
    return ollama_chat(messages)


# ---------------------------------------------------------------------------
# Per-function labeling orchestration
# ---------------------------------------------------------------------------

def label_function(fn: dict) -> dict:
    """Add labels to a single function dict. Returns the mutated dict."""
    code = fn["function_code"]
    call_names = fn.get("call_names", [])

    # Step 1: deterministic side_effects
    side_effects, is_ambiguous = deterministic_side_effects(call_names)

    # Step 2: LLM fallback only if ambiguous
    if is_ambiguous:
        log.info("  → LLM fallback for side_effects (ambiguous): %s", fn.get("function_name"))
        side_effects = llm_side_effects_fallback(code)
        fn["side_effects_source"] = "llm_fallback"
    else:
        fn["side_effects_source"] = "deterministic"

    # Step 3: always LLM for high_level_purpose (2-turn CoT)
    high_level_purpose = llm_high_level_purpose(code)

    # Preserve control_flow_elements from extract.py if present
    control_flow = fn.get("control_flow_elements", [])

    fn["labels"] = {
        "high_level_purpose": high_level_purpose,
        "control_flow_elements": control_flow,
        "side_effects": side_effects if side_effects else ["none"],
    }
    return fn


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def write_checkpoint(functions: list[dict], output_path: Path) -> None:
    """Atomically write current state to output file via a tmp swap."""
    tmp_path = output_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for fn in functions:
            f.write(json.dumps(fn) + "\n")
    tmp_path.replace(output_path)
    log.info("Checkpoint saved → %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Label functions with side_effects and high_level_purpose")
    parser.add_argument("--input",      required=True, help="Path to input functions.jsonl")
    parser.add_argument("--output",     required=True, help="Path to output functions.jsonl")
    parser.add_argument("--limit",      type=int, default=None, help="Process only first N functions (for debugging)")
    parser.add_argument("--resume",     action="store_true", help="Skip functions that already have labels")
    args = parser.parse_args()

    input_path  = Path(args.input)
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

    if args.limit:
        functions = functions[: args.limit]
        log.info("Limited to first %d functions", args.limit)

    total        = len(functions)
    results      = list(functions)   # preserves order; updated by index as workers finish
    results_lock = threading.Lock()
    count_lock   = threading.Lock()
    completed_count = 0
    skipped_count   = 0

    def process_one(i: int, fn: dict):
        if args.resume and "labels" in fn:
            return i, fn, True   # (index, result, was_skipped)

        log.info("[%d/%d] Labeling: %s", i + 1, total, fn.get("function_name", "?"))
        t0 = time.time()
        try:
            labeled = label_function(fn)
        except Exception as exc:
            log.error("  ERROR labeling %s: %s", fn.get("function_name"), exc)
            fn["labels"] = {
                "high_level_purpose": "ERROR",
                "control_flow_elements": fn.get("control_flow_elements", []),
                "side_effects": ["none"],
            }
            fn["side_effects_source"] = "error"
            labeled = fn

        elapsed = time.time() - t0
        log.info(
            "  done in %.1fs  side_effects=%s  source=%s",
            elapsed,
            labeled.get("labels", {}).get("side_effects"),
            labeled.get("side_effects_source"),
        )
        return i, labeled, False

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_one, i, fn): i for i, fn in enumerate(functions)}

        for future in as_completed(futures):
            try:
                i, fn, was_skipped = future.result()
            except Exception as exc:
                log.error("Unexpected worker error: %s", exc)
                continue

            with results_lock:
                results[i] = fn

            with count_lock:
                if was_skipped:
                    skipped_count += 1
                else:
                    completed_count += 1
                current = completed_count

            # Checkpoint every CHECKPOINT_EVERY completed (non-skipped) functions
            if not was_skipped and current % CHECKPOINT_EVERY == 0:
                with results_lock:
                    snapshot = list(results)
                write_checkpoint(snapshot, output_path)
                log.info("  [%d/%d functions completed]", current, total)

    if skipped_count:
        log.info("Skipped %d already-labeled functions (--resume)", skipped_count)

    # Final write
    write_checkpoint(results, output_path)
    log.info("Final output written to %s", output_path)

    # Summary stats
    sources: dict[str, int] = {}
    for fn in results:
        src = fn.get("side_effects_source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    log.info("Labeling complete. side_effects_source breakdown: %s", sources)


if __name__ == "__main__":
    main()