#!/usr/bin/env python3
"""
Colab/T4-friendly GPU-first KG web prototype (PyVis visualization):
 - Quantized model loading (8-bit via bitsandbytes) and device_map="auto"
 - Nemotron-Mini-4B-Instruct preferred; falls back if unavailable
 - Token-chunking, strict TOON extraction, GPU-side aggregation with a single CPU handoff
 - Single worker thread that initializes the model inside the worker (avoids cross-thread CUDA context issues)
 - PyVis for local, offline HTML visualization (no CDN). We create small Top-K projections for safety.
Notes:
 - Requirements (Colab): transformers, torch, bitsandbytes, accelerate, polars, xxhash, flask, pyvis, networkx
   Install once: !pip install -q transformers torch bitsandbytes accelerate polars xxhash flask pyvis networkx
 - Set env var to override model: %env KG_MODEL=nvidia/Nemotron-Mini-4B-Instruct
 - Keep BATCH_SIZE small (1-2) on T4.
"""
import os
import time
import uuid
import re
import threading
import queue
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import xxhash
import polars as pl

# Visualization libs (PyVis + NetworkX)
import networkx as nx
from pyvis.network import Network

from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import (
    Flask, request, redirect, url_for, render_template_string,
    send_from_directory, jsonify, flash
)

# --------------------
# Config (tune these)
# --------------------
APPDIR = Path.cwd()  # Safe for notebook / Colab
OUT_DIR = APPDIR / "tmp_gpu_kg"
UPLOAD_DIR = OUT_DIR / "uploads"
JSON_DIR = OUT_DIR / "json"
HTML_DIR = OUT_DIR / "html"
for d in (OUT_DIR, UPLOAD_DIR, JSON_DIR, HTML_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Preferred model (user requested Nemotron-Mini-4B-Instruct). If not available we'll try fallbacks.
MODEL_NAME = os.getenv("KG_MODEL", "nvidia/Nemotron-Mini-4B-Instruct")
FALLBACK_MODELS = [
    # "mistralai/Mistral-7B-Instruct-V0.2",
    # "mistralai/Mistral-7B-Instruct",
    "philschmid/phi-2-mini"
    
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Conservative defaults for Colab T4 - you can tweak these down for extra safety
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "1024"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "96"))
USE_TORCH_COMPILE = os.getenv("KG_TORCH_COMPILE", "1") == "1"

# Visualization projection limits for PyVis (keeps browser & Colab happy)
MAX_VIS_NODES = 500
MAX_VIS_EDGES = 2000

# Prompt template (TOON)
PROMPT_TEMPLATE = """Extract factual triples from the input. Output using TOON lines only.
Each triple must be on its own line like:
TRIPLE source:"<entity1>" predicate:"<relation>" target:"<entity2>" confidence:<float>
Return ONLY TRIPLE lines. No commentary.

Input:
---
{text}
---
"""

# TOON parsing regexes
_TRIPLE_LINE_RE = re.compile(r'^\s*TRIPLE\s+(.*)$', re.IGNORECASE)
_KEY_VAL_RE = re.compile(r'(\w+)\s*:\s*"([^"]+)"|(\w+)\s*:\s*([\S]+)')

def parse_toon_block(text: str) -> List[Tuple[str,str,str,float]]:
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

# stable 64-bit hash for mapping strings -> int keys (CPU)
def hash64(s: str) -> int:
    return xxhash.xxh64(s, seed=0).intdigest()

# --------------------
# Model loading utility (quantized & robust)
# --------------------
def load_model_safe(model_name: str):
    """
    Try to load model_name in a Colab/T4-friendly way:
      - Prefer 8-bit (bitsandbytes) quantization and device_map="auto"
      - Use torch_dtype=float16 where possible
    Returns (tokenizer, model, actual_model_name)
    Raises Exception if no model can be loaded.
    """
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    candidates = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]
    last_exc = None
    for name in candidates:
        try:
            print(f"[loader] trying model: {name}")
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    name,
                    load_in_8bit=True,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            model.eval()
            print(f"[loader] loaded {name} (quantized, device_map=auto)")
            return tokenizer, model, name
        except Exception as e:
            print(f"[loader] failed to load {name}: {e}")
            last_exc = e
    raise RuntimeError("Failed to load any model. Last error: " + repr(last_exc))

# --------------------
# GPU aggregation (keep tensors GPU-side as long as possible)
# --------------------
def aggregate_triples_gpu(batch_id_arrays: List[Tuple[List[int], List[int], List[int], List[float]]], device: torch.device):
    """
    batch_id_arrays: list of (src_h_list, dst_h_list, rel_h_list, conf_list)
    Returns: list of aggregated tuples (src_h, dst_h, rel_h, mean_conf, count)
    """
    # Build concatenated tensors on GPU
    src_tensors = []
    dst_tensors = []
    rel_tensors = []
    conf_tensors = []
    for srcs, dsts, rels, confs in batch_id_arrays:
        if not srcs:
            continue
        t_src = torch.tensor(srcs, dtype=torch.int64, device=device)
        t_dst = torch.tensor(dsts, dtype=torch.int64, device=device)
        t_rel = torch.tensor(rels, dtype=torch.int64, device=device)
        t_conf = torch.tensor(confs, dtype=torch.float32, device=device)
        src_tensors.append(t_src)
        dst_tensors.append(t_dst)
        rel_tensors.append(t_rel)
        conf_tensors.append(t_conf)

    if len(src_tensors) == 0:
        return []

    src_all = torch.cat(src_tensors, dim=0)
    dst_all = torch.cat(dst_tensors, dim=0)
    rel_all = torch.cat(rel_tensors, dim=0)
    conf_all = torch.cat(conf_tensors, dim=0)

    # Composite key mixing
    k1 = (src_all * 1315423911) ^ (dst_all * 2654435761) ^ (rel_all * 97531)
    keys = k1

    unique_keys, inv = torch.unique(keys, return_inverse=True)
    num_unique = unique_keys.size(0)

    sum_conf = torch.zeros(num_unique, dtype=torch.float32, device=device)
    counts = torch.zeros(num_unique, dtype=torch.int64, device=device)

    sum_conf = sum_conf.scatter_add(0, inv, conf_all)
    counts = counts.scatter_add(0, inv, torch.ones_like(inv, dtype=torch.int64, device=device))

    mean_conf = sum_conf / counts.to(torch.float32)

    # Move minimal arrays back to CPU to build representative mapping
    inv_cpu = inv.cpu().numpy()
    src_cpu = src_all.cpu().numpy()
    dst_cpu = dst_all.cpu().numpy()
    rel_cpu = rel_all.cpu().numpy()
    mean_conf_cpu = mean_conf.cpu().numpy()
    counts_cpu = counts.cpu().numpy()

    first_idx = {}
    for idx, u in enumerate(inv_cpu):
        if u not in first_idx:
            first_idx[u] = idx

    aggregated = []
    for u in range(len(unique_keys)):
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
# Worker thread (single GPU worker)
# --------------------
JOB_QUEUE = queue.Queue()
RESULTS: Dict[str, Optional[str]] = {}  # gid -> html or json path, None on failure

def gpu_worker_loop():
    """
    Single worker that lazily loads the model (inside the worker thread),
    so CUDA context is owned by this thread and avoid cross-thread CUDA issues.
    """
    tokenizer = None
    model = None
    model_name_used = None

    while True:
        job = JOB_QUEUE.get()
        if job is None:
            break
        gid, doc_path = job
        RESULTS[gid] = "processing"
        t0 = time.time()
        try:
            # lazy model init
            if model is None:
                tokenizer, model, model_name_used = load_model_safe(MODEL_NAME)
                if USE_TORCH_COMPILE:
                    try:
                        model = torch.compile(model, mode="reduce-overhead")
                        print("[worker] model compiled with torch.compile")
                    except Exception as e:
                        print("[worker] torch.compile skipped:", e)

            raw = doc_path.read_text(encoding="utf-8", errors="replace")
            # chunk tokenwise to avoid splitting tokens
            enc_all = tokenizer.encode(raw, add_special_tokens=False)
            chunks = []
            i = 0
            while i < len(enc_all):
                j = min(i + CHUNK_TOKENS, len(enc_all))
                chunks.append(tokenizer.decode(enc_all[i:j], skip_special_tokens=True))
                i = j

            batch_hash_arrays = []
            hash_to_text: Dict[int,str] = {}

            # process in small batches, generate greedily (do_sample=False) to be deterministic
            for bi in range(0, len(chunks), BATCH_SIZE):
                batch_chunks = chunks[bi:bi+BATCH_SIZE]
                prompts = [PROMPT_TEMPLATE.format(text=c) for c in batch_chunks]
                # encode inputs and move to model device
                enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                with torch.no_grad():
                    outs = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                # decode outputs
                texts = [tokenizer.decode(o, skip_special_tokens=True) for o in outs]
                for out_text in texts:
                    triples = parse_toon_block(out_text)
                    if not triples:
                        continue
                    src_h, dst_h, rel_h, confs = [], [], [], []
                    for s,p,o,conf in triples:
                        hs = hash64(s); hr = hash64(p); ho = hash64(o)
                        src_h.append(hs); dst_h.append(ho); rel_h.append(hr); confs.append(float(conf))
                        hash_to_text.setdefault(hs, s)
                        hash_to_text.setdefault(hr, p)
                        hash_to_text.setdefault(ho, o)
                    batch_hash_arrays.append((src_h, dst_h, rel_h, confs))

                # free CUDA memory for safety between batches
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # aggregate on GPU
            aggregated = aggregate_triples_gpu(batch_hash_arrays, device=model.device if model is not None else DEVICE)

            # single CPU handoff: convert aggregated rows to human strings
            rows = []
            for src_h, dst_h, rel_h, mean_conf, count in aggregated:
                s = hash_to_text.get(src_h, str(src_h))
                p = hash_to_text.get(rel_h, str(rel_h))
                o = hash_to_text.get(dst_h, str(dst_h))
                rows.append({"source": s, "predicate": p, "target": o, "confidence": float(mean_conf), "count": int(count)})

            if rows:
                df = pl.DataFrame(rows)
            else:
                df = pl.DataFrame([], schema=["source","predicate","target","confidence","count"])

            # compute PageRank on CPU via small networkx projection
            pr_df = pl.DataFrame([])
            try:
                Gnx = nx.DiGraph()
                for r in df.iter_rows(named=True):
                    Gnx.add_edge(r["source"], r["target"], weight=int(r["count"]))
                # If no edges, pagerank will raise; guard
                if Gnx.number_of_nodes() > 0 and Gnx.number_of_edges() > 0:
                    pr = nx.pagerank(Gnx)
                    pr_rows = [{"node": n, "pagerank": float(s)} for n,s in pr.items()]
                    pr_df = pl.DataFrame(pr_rows)
            except Exception:
                pr_df = pl.DataFrame([])

            # Build small projection for PyVis: choose top-K nodes by pagerank (safety)
            top_nodes = set()
            if not pr_df.is_empty():
                k = min(MAX_VIS_NODES, pr_df.height)
                top_nodes = set(pr_df.sort("pagerank", reverse=True).head(k).get_column("node").to_list())
            else:
                # fallback: use highest-degree nodes if no pagerank
                if df.height > 0:
                    # compute degree counts in polars quickly
                    src_counts = df.groupby("source").agg(pl.col("count").sum().alias("deg"))
                    tgt_counts = df.groupby("target").agg(pl.col("count").sum().alias("deg"))
                    degs = pl.concat([src_counts.rename({"source":"node"}), tgt_counts.rename({"target":"node"})], how="vertical")
                    # sum degs by node
                    degs = degs.groupby("node").agg(pl.col("deg").sum().alias("deg"))
                    k = min(MAX_VIS_NODES, degs.height)
                    top_nodes = set(degs.sort("deg", reverse=True).head(k).get_column("node").to_list())

            # Filter edges to those connecting top_nodes
            if top_nodes:
                filt = df.filter(pl.col("source").is_in(list(top_nodes)) & pl.col("target").is_in(list(top_nodes)))
                # If too many edges after filter, downsample by highest count
                if filt.height > MAX_VIS_EDGES:
                    filt = filt.sort("count", reverse=True).head(MAX_VIS_EDGES)
            else:
                # no top nodes (empty graph) -> keep nothing
                filt = df.head(0)

            # Build NetworkX small graph for visualization and then use PyVis
            small_G = nx.DiGraph()
            for r in filt.iter_rows(named=True):
                s = r["source"]; t = r["target"]
                wt = int(r["count"])
                small_G.add_node(s)
                small_G.add_node(t)
                small_G.add_edge(s, t, weight=wt, predicate=r["predicate"], confidence=r["confidence"])

            # create pyvis Network and save HTML locally (self-contained)
            net = Network(height="800px", width="100%", bgcolor="#0f0f0f", font_color="white", notebook=False)
            net.barnes_hut()
            # add nodes and edges
            deg = dict(small_G.degree())
            for n, attrs in small_G.nodes(data=True):
                size = max(10, 12 + deg.get(n, 0) * 2)
                net.add_node(n, label=n, title=n, size=size)
            for u, v, attrs in small_G.edges(data=True):
                title = attrs.get("predicate", "")
                value = min(6, 1 + attrs.get("weight", 1))
                net.add_edge(u, v, title=title, value=value)

            # Save HTML file for this gid (PyVis writes out HTML that includes JS; it's local file)
            html_path = HTML_DIR / f"{gid}.html"
            net.show(str(html_path))  # writes HTML file

            # Also save a compact JSON metadata for API (optional)
            nodes_meta = [{"id": n, "pagerank": float(pr.get(n, 0)) if 'pr' in locals() else 0} for n in small_G.nodes()]
            links_meta = [{"source": u, "target": v, "predicate": d.get("predicate"), "confidence": d.get("confidence"), "count": d.get("weight")} for u,v,d in small_G.edges(data=True)]
            out = {"id": gid, "uploaded": doc_path.name, "created": time.ctime(), "nodes": nodes_meta, "links": links_meta}
            json_path = JSON_DIR / f"{gid}.json"
            import json as _json
            with open(json_path, "w", encoding="utf-8") as fh:
                _json.dump(out, fh, ensure_ascii=False, indent=2)

            # Mark result as the HTML path
            RESULTS[gid] = str(html_path)
            print(f"[worker] finished {gid} in {time.time()-t0:.2f}s, triples={len(rows)}, vis_nodes={small_G.number_of_nodes()}, vis_edges={small_G.number_of_edges()}")

            # free memory proactively
            del net, small_G, filt, df
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        except Exception as e:
            print("[worker] processing error:", e)
            RESULTS[gid] = None
        finally:
            JOB_QUEUE.task_done()

# Spawn single worker thread (model loaded inside worker)
worker_thread = threading.Thread(target=gpu_worker_loop, daemon=True)
worker_thread.start()

# --------------------
# Flask app (minimal)
# --------------------
app = Flask(__name__)
app.secret_key = "dev-key"

INDEX_HTML = """
<!doctype html>
<title>GPU KG (Colab/T4)</title>
<h2>Upload Document for KG Extraction</h2>
<form method=post enctype=multipart/form-data action="{{ url_for('upload') }}">
  <input type=file name=file required>
  <input type=submit value="Upload & Process">
</form>
<p><a href="{{ url_for('list_graphs') }}">Saved graphs</a></p>
"""

LIST_HTML = """
<!doctype html>
<title>Saved Graphs</title>
<h2>Saved graphs</h2>
<ul>
{% for gid, uploaded, created, status in items %}
  <li>{{ gid }} - {{ uploaded }} - {{ created }} - {{ status }} - <a href="{{ url_for('view_graph', gid=gid) }}">View</a></li>
{% endfor %}
</ul>
<p><a href="{{ url_for('index') }}">Upload</a></p>
"""

# We no longer use unpkg or CDN; PyVis HTML file contains the necessary JS references (default behavior).
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
    RESULTS[gid] = "queued"
    JOB_QUEUE.put((gid, saved))
    return redirect(url_for("list_graphs"))

@app.route("/graphs", methods=["GET"])
def list_graphs():
    items = []
    # HTML results first
    for p in sorted(HTML_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.suffix != ".html": continue
        # try to read corresponding json for metadata
        json_p = JSON_DIR / f"{p.stem}.json"
        uploaded = ""
        created = time.ctime(p.stat().st_mtime)
        if json_p.exists():
            import json as _json
            try:
                d = _json.load(open(json_p, "r", encoding="utf-8"))
                uploaded = d.get("uploaded","")
                created = d.get("created", created)
            except Exception:
                pass
        items.append((p.stem, uploaded, created, "done"))
    # queued jobs
    for qitem in list(JOB_QUEUE.queue):
        items.append((qitem[0], qitem[1].name, "queued", "queued"))
    # failed
    for gid, status in list(RESULTS.items()):
        if status == "queued" or status == "processing":
            continue
        if status is None:
            items.append((gid, "(failed)", time.ctime(), "failed"))
    return render_template_string(LIST_HTML, items=items)

@app.route("/view/<gid>")
def view_graph(gid):
    # wait up to a short timeout for result (simple polling for prototype)
    for _ in range(120):
        if gid in RESULTS and isinstance(RESULTS[gid], str):
            # RESULTS[gid] is path to html
            path = Path(RESULTS[gid])
            if path.exists():
                return send_from_directory(str(HTML_DIR), path.name)
            else:
                # fallback: try HTML_DIR/gid.html
                p = HTML_DIR / f"{gid}.html"
                if p.exists():
                    return send_from_directory(str(HTML_DIR), p.name)
                else:
                    return "Result missing", 500
        elif gid in RESULTS and RESULTS[gid] is None:
            return "Processing failed", 500
        time.sleep(0.5)
    return "Timeout waiting for processing", 504

@app.route("/graph_data/<gid>")
def graph_data(gid):
    # keep compatibility: return the compact json metadata if exists
    p = JSON_DIR / f"{gid}.json"
    if not p.exists():
        return jsonify({"nodes":[], "links":[]})
    import json as _json
    data = _json.load(open(p,"r",encoding="utf-8"))
    return jsonify({"nodes": data.get("nodes", []), "links": data.get("links", [])})

@app.route("/uploads/<path:filename>")
def download_uploaded(filename):
    return send_from_directory(str(UPLOAD_DIR), filename, as_attachment=True)

@app.route("/delete/<gid>")
def delete_graph(gid):
    p_html = HTML_DIR / f"{gid}.html"
    if p_html.exists(): p_html.unlink()
    p_json = JSON_DIR / f"{gid}.json"
    if p_json.exists(): p_json.unlink()
    for f in UPLOAD_DIR.iterdir():
        if f.name.startswith(gid):
            f.unlink()
    RESULTS.pop(gid, None)
    flash("Deleted")
    return redirect(url_for("list_graphs"))

if __name__ == "__main__":
    # Flask threaded=False to avoid concurrent requests causing simultaneous GPU use
    print("Starting app on http://127.0.0.1:5000 (threaded=False)")
    app.run(debug=True, port=5000, host="0.0.0.0", threaded=False)
