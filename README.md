# Hallucination Heatmap

**Sentence-level RAG hallucination detection via embedding distance heatmap.**

## Problem

RAG systems hallucinate. Existing tools show overall citation quality (e.g., "this answer has 80% citation coverage"), but they don't tell you *which specific sentences* are grounded vs. fabricated. You get a single aggregate score, not per-sentence granularity.

## Why This Is Different

| Tool | Approach | Limitation |
|------|----------|------------|
| **RAGAS / TruLens / DeepEval** | LLM-as-judge or aggregate metrics | No sentence-level embedding distance; requires LLM calls |
| **LangChain / LlamaIndex evaluators** | Faithfulness, answer relevance | Binary or coarse-grained; not per-sentence heatmap |
| **This tool** | **Embedding cosine distance per sentence → cited chunk** | **Green/red heatmap per sentence; runs locally; no LLM calls** |

The one genuinely new piece: **per-sentence embedding distance visualization** — compute cosine distance between each generated sentence's embedding and its cited source chunk's embedding, render as a color-coded heatmap. No LLM judge, no API calls, runs in seconds on CPU.

## How It Works

```
Generated Text + Citations (JSON)
         │
         ▼
┌─────────────────────────────────────┐
│ 1. Split generated text into sentences         │
│ 2. Map each sentence to its cited chunk        │
│ 3. Batch-embed all sentences + chunks          │
│    (sentence-transformers/all-MiniLM-L6-v2)    │
│ 4. Cosine distance = 1 - (sentence · chunk)    │
│ 5. Threshold → grounded / hallucinated         │
│ 6. Render Rich terminal heatmap                │
└─────────────────────────────────────┘
```

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast, CPU-friendly)
- **Distance metric**: Cosine distance (1 - cosine similarity) on normalized embeddings
- **Threshold**: Default 0.4 (tunable via `--threshold`)
- **Output**: Terminal heatmap (Rich) or JSON (`--json`)

## Prerequisites

- Python 3.10+
- No API keys, no external services — runs fully offline after model download (~90 MB on first run)

## Quick Start

```bash
# 1. Install (CPU-only PyTorch first to avoid CUDA bloat on Linux)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Run demo (built-in sample data)
hallucination-heatmap demo

# 3. Or analyze your own data
hallucination-heatmap analyze sample_input.json

# 4. JSON output for pipelines
hallucination-heatmap analyze sample_input.json --json
```

## Example Output

```
$ hallucination-heatmap demo

╭─────────────────────── Hallucination Heatmap Summary ────────────────────────╮
│ Model:               sentence-transformers/all-MiniLM-L6-v2                  │
│ Threshold:           0.40                                                    │
│ Total sentences:     8                                                       │
│ Cited sentences:     8                                                       │
│ Grounded:            7                                                       │
│ Likely hallucinated: 1                                                       │
│ Errors:              0                                                       │
│ Grounded ratio:      87.5%                                                   │
╰──────────────────────────────────────────────────────────────────────────────╯

┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳
┃   # ┃ Sentence                                 ┃   Distance ┃     Status     ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇
│   1 │ The transformer architecture was         │      0.282 │    GROUNDED    │
│     │ introduced in the 2017 paper 'Attention  │            │                │
│     │ Is All You Need' by Vaswani et al.       │            │                │
│   2 │ It replaced recurrence with              │      0.299 │    GROUNDED    │
│     │ self-attention mechanisms, enabling      │            │                │
│     │ parallel training.                       │            │                │
│   3 │ The model uses multi-head attention to   │      0.246 │    GROUNDED    │
│     │ capture different representation         │            │                │
│     │ subspaces.                               │            │                │
│   4 │ Positional encodings are added to give   │      0.370 │    GROUNDED    │
│     │ the model sequence order information.    │            │                │
│   5 │ The original transformer had 6 encoder   │      0.155 │    GROUNDED    │
│     │ and 6 decoder layers.                    │            │                │
│   6 │ BERT later adapted the encoder-only      │      0.327 │    GROUNDED    │
│     │ architecture for pre-training.           │            │                │
│   7 │ GPT uses a decoder-only architecture     │      0.248 │    GROUNDED    │
│     │ with causal masking.                     │            │                │
│   8 │ Transformers achieve state-of-the-art    │      0.503 │  HALLUCINATED  │
│     │ results on translation benchmarks.       │            │                │
└─────┴──────────────────────────────────────────┴────────────┴────────────────┘

╭─────────────────────────────────── Legend ───────────────────────────────────╮
│ ■  Well grounded (distance ≤ threshold/2)                                    │
│ ■  Moderately grounded (distance ≤ threshold)                                │
│ ■  Weakly grounded (distance ≤ 1.5×threshold)                                │
│ ■  Likely hallucinated (distance > 1.5×threshold)                            │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Sentence 8 is correctly flagged as hallucinated — the citation claims transformers were designed for image classification, but the generated sentence says they achieve SOTA on translation.

## Input Format

```json
{
  "generated_text": "Full generated answer text...",
  "citations": [
    {
      "sentence_idx": 0,
      "chunk_id": "unique_chunk_id",
      "chunk_text": "Source text chunk that supports sentence 0"
    }
  ]
}
```

- `sentence_idx`: 0-based index of the sentence in `generated_text` (after sentence splitting)
- `chunk_id`: Any unique identifier for the source chunk
- `chunk_text`: The actual source text to compare against

## Tech Stack & Libraries Reused

| Library | Purpose | Why |
|---------|---------|-----|
| **sentence-transformers** | Embeddings | Standard, well-maintained, supports many models |
| **numpy** | Cosine distance | Fast vectorized ops, no reinvention |
| **Rich** | Terminal rendering | Beautiful tables, colors, panels — zero config |
| **Typer** | CLI | Type-safe, auto-help, modern |
| **Pydantic** | Data validation | Runtime validation, clear schemas |

## Known Limitations / What's Next

- **Sentence splitting** is regex-based — may mis-split abbreviations (e.g., "e.g.", "Dr.")
- **Single citation per sentence** — doesn't handle multiple citations for one sentence
- **Threshold is global** — could be calibrated per-domain
- **No semantic chunk alignment** — assumes citation `sentence_idx` matches split sentences
- **CPU only** — add `--device cuda` for GPU acceleration
- **No web UI** — could wrap in FastAPI + simple frontend
- **Batch file processing** — add `--input-dir` for folder of JSON files

## License

MIT