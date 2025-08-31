#!/usr/bin/env python3
"""
Local KG web app:
 - Upload a doc (txt/pdf/docx)
 - Extract text (spaCy)
 - Build simple KG (NER + SVO + co-occurrence)
 - Render interactive graph (pyvis) saved to ./tmp_graphs/html/
 - Uploaded files saved to ./tmp_graphs/uploads/
 - Manage saved graphs via /graphs
Note: Not production-grade. Keep local.
"""
from flask import Flask, request, render_template_string, redirect, url_for, send_from_directory, flash
from pathlib import Path
import uuid
import time
import os
import networkx as nx
import spacy
from pyvis.network import Network

# Optional readers
try:
    import PyPDF2
except Exception:
    PyPDF2 = None
try:
    import docx
except Exception:
    docx = None

APPDIR = Path(__file__).resolve().parent
OUT_DIR = APPDIR / "tmp_graphs"
UPLOAD_DIR = OUT_DIR / "uploads"
HTML_DIR = OUT_DIR / "html"
for d in (OUT_DIR, UPLOAD_DIR, HTML_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = "dev-key"  # local only

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")


INDEX_HTML = """
<!doctype html>
<title>Local KG Builder</title>
<h2 style="font-family:Arial">Local Knowledge Graph Builder</h2>
<form method=post enctype=multipart/form-data action="{{ url_for('upload') }}">
  <label>Upload a document (.txt .pdf .docx):</label><br>
  <input type=file name=file required>
  <input type=submit value="Build KG">
</form>
<p>
<a href="{{ url_for('list_graphs') }}">View saved graphs</a>
</p>
<hr>
<p style="color:gray;font-size:0.9em">Files saved to: {{ out_dir }}</p>
"""

LIST_HTML = """
<!doctype html>
<title>Saved Graphs</title>
<h2 style="font-family:Arial">Saved Graphs</h2>
<p><a href="{{ url_for('index') }}">&larr; Upload another document</a></p>
{% if items %}
  <ul>
  {% for name, uploaded, htmlfile, created in items %}
    <li>
      <strong>{{ name }}</strong>
      &nbsp;[uploaded: {{ uploaded }}]
      &nbsp;[created: {{ created }}]
      &nbsp;<a href="{{ url_for('view_html', filename=htmlfile) }}" target="_blank">Open</a>
      &nbsp;<a href="{{ url_for('download_uploaded', filename=uploaded) }}" target="_blank">Download Upload</a>
      &nbsp;<a href="{{ url_for('delete_graph', id=name) }}" onclick="return confirm('Delete this graph and uploaded file?')">Delete</a>
    </li>
  {% endfor %}
  </ul>
{% else %}
  <p>No saved graphs yet.</p>
{% endif %}
<hr>
<p style="color:gray;font-size:0.9em">Temporary files folder: {{ out_dir }}</p>
"""

def read_text_from_path(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".txt":
        return path.read_text(encoding="utf8", errors="ignore")
    if suf == ".pdf" and PyPDF2:
        texts = []
        with open(path, "rb") as f:
            r = PyPDF2.PdfReader(f)
            for p in r.pages:
                t = p.extract_text()
                if t:
                    texts.append(t)
        return "\n".join(texts)
    if suf == ".docx" and docx:
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    # fallback
    return path.read_text(encoding="utf8", errors="ignore")

def build_graph(text: str):
    doc = nlp(text)
    G = nx.DiGraph()

    # Add entities
    for ent in doc.ents:
        # key = ent.text.strip()
        key = str(ent.text).strip()
        if len(key) > 1:
            if not G.has_node(key):
                G.add_node(key, type=ent.label_, size=18)

    # co-occurrence edges in same sentence
    for sent in doc.sents:
        sent_entities = [e.text.strip() for e in sent.ents if len(e.text.strip()) > 1]
        for i in range(len(sent_entities)):
            for j in range(i+1, len(sent_entities)):
                a, b = sent_entities[i], sent_entities[j]
                if not G.has_node(a): G.add_node(a, type="UNK", size=12)
                if not G.has_node(b): G.add_node(b, type="UNK", size=12)
                if G.has_edge(a,b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a,b, relationship="co_occurs", weight=1)

    # simple SVO edges
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subj = token.text
                verb = token.head
                dobj = None
                for ch in verb.children:
                    if ch.dep_ in ("dobj","obj","pobj","attr"):
                        dobj = ch
                        break
                if dobj:
                    subj_span = next((e.text.strip() for e in sent.ents if subj in e.text), subj)
                    obj_span = next((e.text.strip() for e in sent.ents if dobj.text in e.text), dobj.text)
                    subj_label = str(subj_span).strip()
                    obj_label = str(obj_span).strip()

                    if not G.has_node(subj_label):
                        G.add_node(subj_label, type="UNK", size=12)
                    if not G.has_node(obj_label):
                        G.add_node(obj_label, type="UNK", size=12)

                    rel = verb.lemma_.lower()
                    G.add_edge(subj_label, obj_label, relationship=rel, weight=1)

    return G

def graph_to_pyvis(G: nx.Graph, title="KG", bgcolor="#0f0f0f", font_color="white"):
    net = Network(height="700px", width="100%", bgcolor=bgcolor, font_color=font_color)
    net.barnes_hut()
    net.toggle_physics(True)

    type_color = {
        "PERSON":"#ffffff",
        "ORG":"#b3e5fc",
        "GPE":"#a5d6a7",
        "DATE":"#ffe082",
        "NORP":"#ce93d8",
        "PRODUCT":"#ffab91",
        "UNK":"#9e9e9e"
    }

    deg = dict(G.degree())
    for n, attr in G.nodes(data=True):
        ntype = attr.get("type", "UNK")
        color = type_color.get(ntype, "#9e9e9e")
        size = max(10, int(attr.get("size", 12) + deg.get(n,0)*3))
        net.add_node(n, label=n, title=f"{n} ({ntype})", color=color, size=size)

    for u, v, attr in G.edges(data=True):
        label = attr.get("relationship", "")
        width = min(6, 1 + attr.get("weight", 1))
        net.add_edge(u, v, title=label, value=width)

    return net

@app.route("/")
def index():
    return render_template_string(INDEX_HTML, out_dir=str(OUT_DIR))

import chardet

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    raw_bytes = file.read()

    # Detect encoding
    detected = chardet.detect(raw_bytes)
    encoding = detected["encoding"] or "utf-8"

    # Decode safely (replace broken characters instead of crashing)
    text = raw_bytes.decode(encoding, errors="replace")
    orig_name = file.filename

    # 1. Build graph
    G = build_graph(text)

    # 2. Convert to PyVis network
    net = graph_to_pyvis(G, title=orig_name)

    # 3. Generate HTML in memory instead of saving
    html_content = net.generate_html()

    return html_content

@app.route("/graphs")
def list_graphs():
    items = []
    # list html files and corresponding uploads
    for html_file in sorted(HTML_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not html_file.suffix == ".html":
            continue
        uid = html_file.stem
        # find uploaded file that starts with uid
        uploaded = next((p.name for p in UPLOAD_DIR.iterdir() if p.name.startswith(uid)), None)
        created = time.ctime(html_file.stat().st_mtime)
        items.append((uid, uploaded, html_file.name, created))
    return render_template_string(LIST_HTML, items=items, out_dir=str(OUT_DIR))

@app.route("/view/<path:filename>")
def view_html(filename):
    return send_from_directory(str(HTML_DIR), filename)

@app.route("/uploaded/<path:filename>")
def download_uploaded(filename):
    return send_from_directory(str(UPLOAD_DIR), filename, as_attachment=True)

@app.route("/delete/<id>")
def delete_graph(id):
    # delete html and matching upload
    htmlf = HTML_DIR / f"{id}.html"
    if htmlf.exists():
        htmlf.unlink()
    for p in UPLOAD_DIR.iterdir():
        if p.name.startswith(id):
            p.unlink()
    flash("Deleted")
    return redirect(url_for("list_graphs"))

if __name__ == "__main__":
    print("Starting local KG web app. Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000)
