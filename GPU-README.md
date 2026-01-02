# GPU-First Knowledge Graph Builder (Colab / T4 Friendly)

This project is an **experimental, GPU-first Knowledge Graph (KG) extraction and visualization system** designed to run safely on **Google Colab with an NVIDIA T4 GPU**.

It combines:

* **LLM-based relation extraction** (Nemotron-Mini-4B-Instruct preferred)
* **Strict TOON (Token-Oriented Object Notation)** for deterministic triple parsing
* **GPU-side batching and aggregation** to minimize CPU↔GPU transfers
* **Quantized inference (8-bit)** to prevent Colab RAM/GPU crashes
* A **minimal Flask web interface** for uploading documents and viewing graphs

The system is intentionally **research-grade**, not production-grade.

---

## High-Level Architecture

```
Document Upload
     ↓
Token-wise Chunking
     ↓
LLM (Quantized, GPU) → TOON triples
     ↓
GPU-side Triple Hashing & Aggregation
     ↓ (single CPU handoff)
Polars DataFrames + Graph Stats
     ↓
Graph JSON
     ↓
Browser Visualization
```

### Key design goals

* Keep **LLM inference and triple aggregation on GPU**
* Avoid repeated CPU↔GPU memory transfers
* Prevent Colab OOM crashes
* Maintain deterministic, parseable outputs (no free-form JSON)

---

## Features

* 📄 Upload `.txt` documents
* 🧠 LLM-based factual triple extraction
* 📐 TOON grammar (machine-parsable, line-based)
* 🚀 Quantized LLM loading (bitsandbytes, 8-bit)
* 🔁 Batch processing with GPU memory cleanup
* 📊 Optional PageRank computation
* 🌐 Web UI for viewing extracted graphs
* 🧵 Single GPU worker thread (safe CUDA context handling)

---

## Model Strategy

### Preferred

* **`nvidia/Nemotron-Mini-4B-Instruct`**

  * Good reasoning quality
  * Fits on T4 **only with 8-bit quantization**
  * Deterministic extraction with `do_sample=False`

### Automatic fallbacks

If Nemotron cannot load:

* `mistralai/Mistral-7B-Instruct`
* `TinyLlama/TinyLlama-1.1B-Chat`
* Lightweight Phi variants

Model selection is automatic and robust.

---

## TOON Output Format

The LLM is forced to output **only** this format:

```
TRIPLE source:"Entity A" predicate:"relation" target:"Entity B" confidence:0.91
```

Why TOON?

* No JSON hallucinations
* Streaming-friendly
* Easy to parse line-by-line
* Works well with batching

---

## Installation (Google Colab)

```bash
pip install -q \
  transformers \
  torch \
  bitsandbytes \
  accelerate \
  polars \
  xxhash \
  flask
```

> ⚠️ Do **not** load models in full FP32 on Colab.
> Quantization is mandatory.

---

## Running the App

In Colab:

```python
!python app.py
```

Since Colab does **not expose `localhost`**, use:

* **ngrok** (recommended), or
* Colab’s built-in port proxy

---

## Environment Variables

Optional tuning:

```bash
export KG_MODEL=nvidia/Nemotron-Mini-4B-Instruct
export CHUNK_TOKENS=1024
export BATCH_SIZE=1
export MAX_NEW_TOKENS=96
export KG_TORCH_COMPILE=1
```

Recommended defaults are already safe for T4.

---

## GPU Safety Notes (Very Important)

* **Single GPU worker thread**
* Quantized model (`load_in_8bit=True`)
* Small batch sizes
* Explicit `torch.cuda.empty_cache()` between batches
* No concurrent Flask GPU requests

This is deliberate. Removing these will crash Colab.

---

## What This Project Is NOT

* ❌ Not a Neo4j replacement
* ❌ Not a production KG pipeline
* ❌ Not async-optimized (by design)

---

## Planned Extensions

* Graphistry integration (streamed visualization)
* cuGraph-based analytics (when available)
* Disk-backed triple stores
* Async FastAPI frontend
* Multi-document KG merging
* Browser-side WebGPU rendering

---

## Who This Is For

* NLP / KG researchers
* GPU-constrained experimentation
* Students exploring LLM-based knowledge extraction
* Engineers prototyping KG pipelines on Colab

---

## License

Open-source, research use encouraged.
Model licenses apply separately.

---


Just say where you want to push this system next.
