#!/usr/bin/env python3
"""
GPU-first KG web prototype (revised):
 - Upload doc
 - LLM (GPU) extracts TOON triples (batch prompts)
 - Triples -> 64-bit hashes -> moved to GPU in batches
 - GPU-only aggregation (dedupe + mean confidence + count) using PyTorch tensors
 - Single CPU handoff: aggregated edges -> polars DataFrame -> saved JSON for web
 - Frontend renders with WebGL Force-Graph (three.js / 3d-force-graph)
Notes:
 - This revision removes the extra second-model pass and preserves hash->text mapping
 - Default model is set to an NVIDIA Nemotron-like identifier (override with KG_MODEL env var)
 - torch-tensorrt compile kept optional
 - Requires: torch, transformers, polars, xxhash, (optional torch_tensorrt)
"""
import os
import time
import uuid
import re
import threading
import queue
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import xxhash
import polars as pl

from flask import (
    Flask, request, redirect, url_for, render_template_string,
    send_from_directory, jsonify, flash
)

# --------------------
# Configuration
# --------------------
APPDIR = Path(__file__).resolve().parent
OUT_DIR = APPDIR / "tmp_gpu_kg"
UPLOAD_DIR = OUT_DIR / "uploads"
JSON_DIR = OUT_DIR / "json"
for d in (OUT_DIR, UPLOAD_DIR, JSON_DIR):
    d.mkdir(parents=True, exist_ok=True)


# MODEL_NAME = os.getenv("KG_MODEL", "nvidia/Nemotron-Mini-4B-Instruct")
MODEL_NAME = "nvidia/Nemotron-Mini-4B-Instruct"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "1500"))
CHUNK_TOKENS = 1500
# BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
BATCH_SIZE = 4
# MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
MAX_NEW_TOKENS = 128
USE_TRT = os.getenv("KG_USE_TRT", "0") == "1"

PROMPT_TEMPLATE = """
Extract factual triples from the input. Output using TOON lines only.
Format each triple on its own line exactly like:
TRIPLE source:"<entity1>" predicate:"<relation>" target:"<entity2>" confidence:<float>
Return only TRIPLE lines and nothing else.

Input:
---
{text}
---
"""

# TOON parsing regexes
_TRIPLE_LINE_RE = re.compile(r'^\s*TRIPLE\s+(.*)$', re.IGNORECASE)
_KEY_VAL_RE = re.compile(r'(\w+)\s*:\s*"([^"]+)"|(\w+)\s*:\s*([\S]+)')

def parse_toon_block(text: str) -> List[Tuple[str,str,str,float]]:
    """Parse TOON block and return list of (s,p,o,conf)."""
    triples = []
    for line in text.splitlines():
        m = _TRIPLE_LINE_RE.match(line)
        if not m:
            continue
        rest = m.group(1)
        data = {}
        for m2 in _KEY_VAL_RE.finditer(rest):
            if m2.group(1) and m2.group(2) is not None:
                k = m2.group(1).lower(); v = m2.group(2)
            else:
                k = m2.group(3).lower(); v = m2.group(4)
            data[k] = v
        s = str(data.get("source","")).strip()
        p = str(data.get("predicate","")).strip()
        o = str(data.get("target","")).strip()
        try:
            conf = float(data.get("confidence", "1.0"))
        except Exception:
            conf = 1.0
        if s and p and o:
            triples.append((s,p,o,conf))
    return triples

# --------------------
# Model wrapper
# --------------------
class LLMWorker:
    def __init__(self, model_name=MODEL_NAME, device=DEVICE, use_tensorrt: bool = USE_TRT):
        self.device = device
        # Trust remote code for vendor models; in controlled env you may set trust_remote_code=False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

        # optional torch-tensorrt compile (best-effort)
        if use_tensorrt:
            try:
                import torch_tensorrt
                # Build a simple input spec for TRT compile; may need adjustment per model
                example = torch.zeros((1, 8), dtype=torch.int64, device=self.device)
                self.model = torch_tensorrt.compile(self.model, inputs=[torch_tensorrt.Input(example)], enabled_precisions={torch.float16})
                print("[LLMWorker] model compiled via torch_tensorrt")
            except Exception as e:
                print("[LLMWorker] torch_tensorrt compile failed:", e)

    @torch.no_grad()
    def generate_batch(self, prompts: List[str], max_new_tokens: int = MAX_NEW_TOKENS) -> List[str]:
        """Generate raw text outputs for a batch of prompts."""
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outs = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        texts = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outs]
        return texts

# --------------------
# Utilities
# --------------------
def hash64(s: str) -> int:
    """Stable 64-bit hash for a string (used to move identifiers to integer space)."""
    return xxhash.xxh64(s, seed=0).intdigest()

# --------------------
# GPU aggregation
# --------------------
def aggregate_triples_gpu(batch_id_arrays: List[Tuple[List[int], List[int], List[int], List[float]]], device=DEVICE):
    """
    Input: list of tuples per parsed batch: (src_hashes, dst_hashes, rel_hashes, confs)
    Returns: list of aggregated rows (src_hash, dst_hash, rel_hash, mean_conf, count)
    Aggregation performed primarily on GPU; final representative mapping chosen via CPU on unique keys.
    """
    # Build concatenated tensors on GPU
    src_tensors = []
    dst_tensors = []
    rel_tensors = []
    conf_tensors = []
    for srcs, dsts, rels, confs in batch_id_arrays:
        if len(srcs) == 0:
            continue
        t_src = torch.tensor(srcs, dtype=torch.int64, device=device)
        t_dst = torch.tensor(dsts, dtype=torch.int64, device=device)
        t_rel = torch.tensor(rels, dtype=torch.int64, device=device)
        t_conf = torch.tensor(confs, dtype=torch.float32, device=device)
        src_tensors.append(t_src)
        dst_tensors.append(t_dst)
        rel_tensors.append(t_rel)
        conf_tensors.append(t_conf)

    if not src_tensors:
        return []

    src_all = torch.cat(src_tensors, dim=0)
    dst_all = torch.cat(dst_tensors, dim=0)
    rel_all = torch.cat(rel_tensors, dim=0)
    conf_all = torch.cat(conf_tensors, dim=0)

    # Composite key mixing (64-bit signed space)
    k1 = (src_all * 1315423911) ^ (dst_all * 2654435761) ^ (rel_all * 97531)
    keys = k1  # int64 tensor

    # unique keys and inverse mapping
    unique_keys, inv = torch.unique(keys, return_inverse=True)
    num_unique = unique_keys.size(0)

    # aggregate sum(conf) and counts using scatter_add on GPU
    sum_conf = torch.zeros(num_unique, dtype=torch.float32, device=device)
    counts = torch.zeros(num_unique, dtype=torch.int64, device=device)

    sum_conf = sum_conf.scatter_add(0, inv, conf_all)
    counts = counts.scatter_add(0, inv, torch.ones_like(inv, dtype=torch.int64, device=device))

    mean_conf = sum_conf / counts.to(torch.float32)

    # Move necessary arrays to CPU to pick representative tuple per unique key.
    inv_cpu = inv.cpu().numpy()
    src_cpu = src_all.cpu().numpy()
    dst_cpu = dst_all.cpu().numpy()
    rel_cpu = rel_all.cpu().numpy()
    mean_conf_cpu = mean_conf.cpu().numpy()
    counts_cpu = counts.cpu().numpy()

    # Build first-occurrence map for representative index
    first_idx = {}
    for idx, u in enumerate(inv_cpu):
        if u not in first_idx:
            first_idx[u] = idx

    aggregated = []
    # For each unique id u, pick representative index idx0
    for u in range(num_unique):
        if u not in first_idx:
            continue
        idx0 = first_idx[u]
        aggregated.append((
            int(src_cpu[idx0]),
            int(dst_cpu[idx0]),
            int(rel_cpu[idx0]),
            float(mean_conf_cpu[u]),
            int(counts_cpu[u])
        ))

    return aggregated

# --------------------
# Background worker and queues
# --------------------
JOB_QUEUE = queue.Queue()
RESULTS: Dict[str, str] = {}  # gid -> json_path (or None on failure). presence = completed/failed.

def gpu_processing_worker():
    """Singleton GPU worker thread: consumes jobs from JOB_QUEUE"""
    model = None
    tokenizer = None
    while True:
        job = JOB_QUEUE.get()
        if job is None:
            break
        gid, doc_path = job
        try:
            t0 = time.time()
            if model is None:
                model = LLMWorker(MODEL_NAME, DEVICE, use_tensorrt=USE_TRT)
                tokenizer = model.tokenizer

            # read file
            raw = doc_path.read_text(encoding="utf-8", errors="replace")

            # chunk tokenwise using tokenizer to avoid splitting tokens
            enc_all = tokenizer.encode(raw, add_special_tokens=False)
            chunks = []
            i = 0
            while i < len(enc_all):
                j = min(i + CHUNK_TOKENS, len(enc_all))
                chunks.append(tokenizer.decode(enc_all[i:j], skip_special_tokens=True))
                i = j

            # We'll build batch_hash_arrays and a mapping hash->text in one pass (no second model pass)
            batch_hash_arrays = []  # List[(src_h_list, dst_h_list, rel_h_list, conf_list)]
            hash_to_text: Dict[int,str] = {}

            # process in batches
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i+BATCH_SIZE]
                prompts = [PROMPT_TEMPLATE.format(text=c) for c in batch]
                outs = model.generate_batch(prompts, max_new_tokens=MAX_NEW_TOKENS)
                for out in outs:
                    triples = parse_toon_block(out)
                    if not triples:
                        continue
                    src_h = []
                    dst_h = []
                    rel_h = []
                    confs = []
                    for s,p,o,conf in triples:
                        hs = hash64(s); hr = hash64(p); ho = hash64(o)
                        src_h.append(hs); rel_h.append(hr); dst_h.append(ho); confs.append(float(conf))
                        # record mapping once
                        if hs not in hash_to_text:
                            hash_to_text[hs] = s
                        if hr not in hash_to_text:
                            hash_to_text[hr] = p
                        if ho not in hash_to_text:
                            hash_to_text[ho] = o
                    batch_hash_arrays.append((src_h, dst_h, rel_h, confs))

            # GPU aggregation
            aggregated = aggregate_triples_gpu(batch_hash_arrays, device=DEVICE if torch.cuda.is_available() else torch.device("cpu"))

            # Single CPU handoff: convert aggregated rows back to readable strings using hash_to_text
            rows = []
            for src_h, dst_h, rel_h, mean_conf, count in aggregated:
                s = hash_to_text.get(src_h, str(src_h))
                p = hash_to_text.get(rel_h, str(rel_h))
                o = hash_to_text.get(dst_h, str(dst_h))
                rows.append({"source": s, "predicate": p, "target": o, "confidence": float(mean_conf), "count": int(count)})

            if rows:
                df = pl.DataFrame(rows)
            else:
                # empty
                df = pl.DataFrame([], schema=["source","predicate","target","confidence","count"])

            # lightweight PageRank on CPU (NetworkX) - for small/medium graphs this is acceptable
            pr_df = pl.DataFrame([])
            try:
                import networkx as nx
                Gnx = nx.DiGraph()
                for r in df.iter_rows(named=True):
                    Gnx.add_edge(r["source"], r["target"], weight=int(r["count"]))
                pr = nx.pagerank(Gnx)
                pr_rows = [{"node": n, "pagerank": float(s)} for n,s in pr.items()]
                pr_df = pl.DataFrame(pr_rows)
            except Exception:
                pr_df = pl.DataFrame([])

            # nodes + links for frontend
            nodes = {}
            for row in df.iter_rows(named=True):
                nodes.setdefault(row["source"], {"id": row["source"], "type": None, "pagerank": 0})
                nodes.setdefault(row["target"], {"id": row["target"], "type": None, "pagerank": 0})
            for r in pr_df.iter_rows(named=True):
                if r["node"] in nodes:
                    nodes[r["node"]]["pagerank"] = r["pagerank"]

            links = [{"source": r["source"], "target": r["target"], "predicate": r["predicate"], "confidence": r["confidence"], "count": r["count"]} for r in df.iter_rows(named=True)]

            out = {"id": gid, "uploaded": doc_path.name, "created": time.ctime(), "nodes": list(nodes.values()), "links": links}
            json_path = JSON_DIR / f"{gid}.json"
            import json
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=2)

            RESULTS[gid] = str(json_path)
            t1 = time.time()
            print(f"[worker] finished {gid} in {t1-t0:.2f}s, triples={len(rows)}")
        except Exception as e:
            print("Processing error:", e)
            RESULTS[gid] = None
        finally:
            JOB_QUEUE.task_done()

# start worker thread
worker_thread = threading.Thread(target=gpu_processing_worker, daemon=True)
worker_thread.start()

# --------------------
# Flask app + minimal frontend
# --------------------
app = Flask(__name__)
app.secret_key = "dev-key"

INDEX_HTML = """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>GPU KG</title></head>
  <body style="font-family:Arial;margin:20px;">
    <h2>GPU KG Builder (upload)</h2>
    <form method=post enctype=multipart/form-data action="{{ url_for('upload') }}">
      <input type=file name=file required>
      <input type=submit value="Upload & Process">
    </form>
    <p><a href="{{ url_for('list_graphs') }}">Saved graphs</a></p>
  </body>
</html>
"""

LIST_HTML = """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Saved</title></head>
  <body style="font-family:Arial;margin:20px;">
    <h2>Saved graphs</h2>
    <ul>
    {% for gid, uploaded, created, status in items %}
      <li>{{ gid }} - {{ uploaded }} - {{ created }} - {{ status }} - <a href="{{ url_for('view_graph', gid=gid) }}">View</a></li>
    {% endfor %}
    </ul>
    <p><a href="{{ url_for('index') }}">Upload</a></p>
  </body>
</html>
"""

VIEW_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>KG {{ gid }}</title>
    <style>body{margin:0;background:#0f0f0f;color:#fff} #graph{width:100%;height:100vh}</style>
  </head>
  <body>
    <div id="graph"></div>
    <script src="https://unpkg.com/three"></script>
    <script src="https://unpkg.com/3d-force-graph"></script>
    <script>
      async function load() {
        const res = await fetch("/graph_data/{{ gid }}");
        if (!res.ok) { document.body.innerHTML = "<h2>Error loading graph</h2>"; return; }
        const data = await res.json();
        const Graph = ForceGraph3D()(document.getElementById('graph'))
          .graphData(data)
          .nodeLabel(n => `${n.id} (PR:${(n.pagerank||0).toFixed(3)})`)
          .nodeAutoColorBy('type')
          .linkDirectionalArrowLength(4)
          .linkDirectionalArrowRelPos(1)
          .onNodeClick(node => { alert(node.id) });
      }
      load();
    </script>
  </body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        flash("No file")
        return redirect(url_for("index"))
    name = Path(f.filename).name
    gid = str(int(time.time())) + "-" + uuid.uuid4().hex[:8]
    saved = UPLOAD_DIR / f"{gid}-{name}"
    f.save(str(saved))
    # enqueue job and return immediately
    RESULTS[gid] = "queued"
    JOB_QUEUE.put((gid, saved))
    return redirect(url_for("list_graphs"))

@app.route("/graphs", methods=["GET"])
def list_graphs():
    items = []
    # completed
    for p in sorted(JSON_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.suffix != ".json": continue
        import json
        d = json.load(open(p,"r",encoding="utf-8"))
        items.append((p.stem, d.get("uploaded",""), d.get("created",""), "done"))
    # queued
    for qitem in list(JOB_QUEUE.queue):
        items.append((qitem[0], qitem[1].name, "queued", "queued"))
    # worker results with pending/completed/failure statuses
    for gid, status in list(RESULTS.items()):
        if status == "queued":
            continue
        if status is None:
            items.append((gid, "(failed)", time.ctime(), "failed"))
    return render_template_string(LIST_HTML, items=items)

@app.route("/view/<gid>")
def view_graph(gid):
    # wait until result exists (small blocking wait)
    for _ in range(120):
        if gid in RESULTS and isinstance(RESULTS[gid], str):
            return render_template_string(VIEW_HTML, gid=gid)
        elif gid in RESULTS and RESULTS[gid] is None:
            return "Processing failed", 500
        time.sleep(0.5)
    return "Timeout waiting for processing", 504

@app.route("/graph_data/<gid>")
def graph_data(gid):
    p = JSON_DIR / f"{gid}.json"
    if not p.exists():
        return jsonify({"nodes":[], "links":[]})
    import json
    data = json.load(open(p,"r",encoding="utf-8"))
    return jsonify({"nodes": data.get("nodes", []), "links": data.get("links", [])})

@app.route("/uploads/<path:filename>")
def download_uploaded(filename):
    return send_from_directory(str(UPLOAD_DIR), filename, as_attachment=True)

@app.route("/delete/<gid>")
def delete_graph(gid):
    p = JSON_DIR / f"{gid}.json"
    if p.exists(): p.unlink()
    for f in UPLOAD_DIR.iterdir():
        if f.name.startswith(gid):
            f.unlink()
    RESULTS.pop(gid, None)
    flash("Deleted")
    return redirect(url_for("list_graphs"))

if __name__ == "__main__":
    print("Starting app on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
