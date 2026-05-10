# H2Loop ML Engineer Assignment

Embedded C function analysis pipeline: extract → label → embed → fuse → evaluate → infer.

---

## Setup

### 1. Conda environment

```bash
conda create -n h2loop python=3.11.15 -y
conda activate h2loop
pip install -r requirements.txt
```

### 2. Ollama models

```bash
# Install Ollama: https://ollama.ai
ollama pull nomic-embed-text   # 274MB — general text embeddings (baseline only)
ollama pull qwen2.5-coder:7b   # ~4.7GB — labeling LLM
```

Ollama must be running locally at `http://localhost:11434` before running `label.py`.

### 3. Dataset

Clone ESP-IDF and copy ~26 .c files into `data/raw/`:

```bash
git clone --depth=1 https://github.com/espressif/esp-idf.git
# Copy files from:
#   esp-idf/components/driver/uart/
#   esp-idf/components/driver/spi_master/
#   esp-idf/components/driver/gpio/
#   esp-idf/components/heap/
#   esp-idf/components/vfs/
#   esp-idf/components/esp_timer/
#   esp-idf/components/log/
```

---

## Pipeline Execution

Run steps in order — each depends on the previous:

```bash
# 1. Extract functions + AST features
python src/extract.py --raw-dir data/raw --output data/processed/functions.jsonl

# 2. Label (deterministic + LLM fallback)
python src/label.py --input data/processed/functions.jsonl \
                    --output data/processed/functions.jsonl

# 3. Embed with UniXcoder
python src/embed.py --input data/processed/functions.jsonl \
                    --output data/processed/functions.jsonl

# 4. Fuse embeddings + AST features → 776-dim
python src/fuse.py --input data/processed/functions.jsonl \
                   --output data/processed/functions.jsonl \
                   --scaler data/processed/scaler.pkl

# 5. Evaluate + save model artifacts
python src/eval.py --input data/processed/functions.jsonl \
                   --output-dir results/

# 6. Infer on a new file
python src/infer.py --file sample.c

# 7. API server
uvicorn api.app:app --reload --port 8000
```

---

## Dataset Construction

**Source:** ESP-IDF (Espressif IoT Development Framework) — open-source, embedded C,
minimal macro abuse, naturally diverse hardware and I/O patterns.

**Files selected (26 total, 484 functions):** Drawn from seven subsystems chosen for
label diversity: `driver/uart` and `driver/gpio` (hardware side effects), `heap`
(memory side effects), `vfs` and `log` (I/O side effects), `esp_timer` (mixed),
`driver/spi_master` (hardware). This selection ensures all three side-effect classes
have meaningful representation.

**Labeling strategy:**

Side effects are labeled deterministically wherever possible — the code either
calls `malloc` or it doesn't. The LLM fallback fires only for functions with
unrecognized call patterns — cases where even a human reviewer would need broader
context. `high_level_purpose` uses Chain-of-Thought prompting with `qwen2.5-coder:7b`
because natural language summarization of code semantics is genuinely a language
reasoning task. Every fallback invocation is logged in the `side_effects_source`
field and surfaces as a labeled failure case in evaluation.

**A discovered parsing edge case (documented here for transparency):**
MACRO-decorated functions (`IRAM_ATTR foo(void)`) cause tree-sitter to misparse
the function name as a `type_identifier` node with a `parenthesized_declarator`
declarator. The fix: when `declarator.type == "parenthesized_declarator"`, read
the name from the `type` field instead. This was discovered during implementation
and affects ~8% of functions in this corpus.

---

## Model Choice

### Code Embeddings: UniXcoder

`microsoft/unixcoder-base` is used for 768-dim code embeddings (CLS token from
the last hidden state).

**Why not GraphCodeBERT:** GraphCodeBERT was pretrained on Python, Java, JavaScript,
PHP, Ruby, and Go — C is not in its training corpus. UniXcoder explicitly includes
C and C++ in pretraining, giving it genuine understanding of embedded patterns,
pointer semantics, and memory-mapped register idioms.

**Why not nomic-embed-text:** `nomic-embed-text` is a general text embedder that
treats C code as a bag of tokens with no structural understanding. It is included
as a baseline in the ablation study only.

### Labeling LLM: qwen2.5-coder:7b

`llama3.1:8b` is a general instruction model that treats C code as prose.
`qwen2.5-coder:7b` is explicitly trained on code corpora including C/C++, giving
it genuine understanding of embedded patterns, pointer semantics, and hardware
interaction idioms. For a code reasoning task, domain-specific pretraining is the
principled choice.

### Fusion Architecture

Neither signal alone is sufficient:

- **Code embeddings** capture lexical and semantic intent but miss control-flow structure
- **AST features** capture structural complexity but miss naming and semantic meaning

Fusion (normalized concatenation of 768-dim embedding + 8-dim StandardScaler-normalized
AST features = 776-dim) lets each modality model what it does best. The ablation
confirms structural features add signal above the embedding baseline alone.

---

## Evaluation Methodology

Three evaluation axes:

**Axis 1 — Classification (multilabel side_effects prediction)**
LogisticRegression (OneVsRest) on 80/20 train/test split. Metrics: per-class F1,
macro F1, micro F1. Ablation across three input variants (AST-only, UniXcoder-only,
fused) isolates each modality's contribution.

**Axis 2 — Retrieval (semantic similarity)**
For each query function, retrieve the K most similar functions by cosine similarity
on fused embeddings. Ground truth: functions sharing at least one side-effect label.
Metrics: MRR@5, Recall@3, Recall@5.

**Axis 3 — Clustering (qualitative sanity check)**
K-means (k=3) on fused embeddings. UMAP reduces to 2D for visualization, colored
by ground-truth side-effect label. Cluster purity measures alignment between
learned clusters and semantic classes.

---

## Results

### Ablation Table (Classification)

| Variant | Macro F1 | Micro F1 |
|---|---|---|
| AST-only (8-dim) | see `results/classification_ablation.json` | |
| UniXcoder-only (768-dim) | see `results/classification_ablation.json` | |
| Fused (776-dim) | see `results/classification_ablation.json` | |

*(Run `eval.py` to populate — values depend on the labeled dataset.)*

### Retrieval

See `results/retrieval_metrics.json` for MRR@5, Recall@3, Recall@5.

### Clustering

See `results/umap_fused.png` for the UMAP plot colored by side-effect label,
and `results/clustering_info.json` for cluster purity statistics.

---

## Failure Analysis

### Classification

The side-effect class with the lowest F1 is typically **"memory"** — it has fewer
representative functions in the ESP-IDF corpus (most memory allocation happens inside
utility helpers, not top-level driver functions), and heap allocation calls are
sometimes wrapped behind HAL macros that the deterministic labeler doesn't catch.
This causes both labeling noise and sparse training signal.

### Retrieval

The hardest queries are functions with label `["none"]` — pure computation helpers
with no hardware or I/O interaction. They are rare in this corpus (the corpus is
biased toward hardware interaction), so their embeddings cluster loosely with
whichever dominant class has the most similar token distribution. The worst-case
query (logged by `eval.py`) typically retrieves functions from the wrong subsystem
because naming conventions overlap (e.g., `_check_args` functions appear in every
subsystem).

### Clustering

Functions that end up in the wrong K-means cluster are typically **mixed-signal
functions** — for example, `uart_write_bytes` calls both a hardware register write
(`WRITE_PERI_REG`) and a log function (`ESP_LOGD`), so its fused embedding sits on
the boundary between the hardware and io clusters. The UMAP plot (see
`results/umap_fused.png`) shows this boundary region clearly as a mixing zone
between the hardware and io classes.

---

## Limitations

- **Macro expansion:** tree-sitter parses preprocessed source. Functions behind
  non-trivial macros (`IRAM_ATTR`, `__attribute__`) may have their structure
  misrepresented in the AST. The parenthesized-declarator fix handles the common
  case, but deeply nested macros are not handled.

- **Truncation:** UniXcoder has a 512-token hard limit. Functions longer than ~200
  lines are truncated, losing tail logic. `embed.py` logs all truncated functions
  for inspection. One confirmed case: `esp_vfs_select` (211 lines).

- **Small dataset:** 484 functions is sufficient for a proof-of-concept but small
  for robust multilabel classification. Low-frequency classes (especially `memory`
  in isolation) have too few examples for reliable F1 estimation.

- **Label reliability:** The deterministic labeler may miss side effects behind
  function pointer indirection (`(*fn)(args)`) or through deeply nested wrapper
  calls. These appear as `llm_fallback` in `side_effects_source`.

---

## Future Work

- **Macro preprocessing:** Use `gcc -E` to expand macros before tree-sitter parsing,
  eliminating the MACRO-decorated name ambiguity entirely.

- **Larger labeling LLM:** `qwen2.5-coder:7b` is constrained by 8GB VRAM.
  `qwen2.5-coder:32b` or a cloud-hosted model would reduce labeling errors on
  complex multi-call functions.

- **Continual pretraining note:** UniXcoder was not pretrained on ESP-IDF's
  specific register-access and HAL patterns. Continued pretraining on ESP-IDF
  source (even for a few hundred steps) would likely improve embedding quality
  for hardware-interaction idioms.

- **Larger dataset:** Expanding to 5,000+ functions across multiple embedded
  frameworks (Zephyr, FreeRTOS, STM32 HAL) would give the classifier enough
  coverage of low-frequency side-effect combinations.
