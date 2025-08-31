import sys
from pathlib import Path
import chardet
import time
import spacy
import networkx as nx
from pyvis.network import Network

try:
    import PyPDF2
except Exception:
    PyPDF2 = None
try:
    import docx
except Exception:
    docx = None

nlp = spacy.load('en_core_web_sm')

INDEX_HTML = """(unused)"""


def read_text_from_path(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == '.txt':
        return path.read_text(encoding='utf8', errors='ignore')
    if suf == '.pdf' and PyPDF2:
        texts = []
        with open(path, 'rb') as f:
            r = PyPDF2.PdfReader(f)
            for p in r.pages:
                t = p.extract_text()
                if t:
                    texts.append(t)
        return '\n'.join(texts)
    if suf == '.docx' and docx:
        d = docx.Document(path)
        return '\n'.join(p.text for p in d.paragraphs)
    # fallback: try to detect encoding
    raw = path.read_bytes()
    detected = chardet.detect(raw)
    enc = detected.get('encoding') or 'utf-8'
    return raw.decode(enc, errors='replace')


def build_graph(text: str):
    doc = nlp(text)
    G = nx.DiGraph()

    for ent in doc.ents:
        key = str(ent.text).strip()
        if len(key) > 1:
            if not G.has_node(key):
                G.add_node(key, type=ent.label_, size=18)

    for sent in doc.sents:
        sent_entities = [e.text.strip() for e in sent.ents if len(e.text.strip()) > 1]
        for i in range(len(sent_entities)):
            for j in range(i+1, len(sent_entities)):
                a, b = sent_entities[i], sent_entities[j]
                if not G.has_node(a): G.add_node(a, type='UNK', size=12)
                if not G.has_node(b): G.add_node(b, type='UNK', size=12)
                if G.has_edge(a,b):
                    G[a][b]['weight'] += 1
                else:
                    G.add_edge(a,b, relationship='co_occurs', weight=1)

    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ('nsubj', 'nsubjpass'):
                subj = token.text
                verb = token.head
                dobj = None
                for ch in verb.children:
                    if ch.dep_ in ('dobj','obj','pobj','attr'):
                        dobj = ch
                        break
                if dobj:
                    subj_span = next((e.text.strip() for e in sent.ents if subj in e.text), subj)
                    obj_span = next((e.text.strip() for e in sent.ents if dobj.text in e.text), dobj.text)
                    subj_label = str(subj_span).strip()
                    obj_label = str(obj_span).strip()

                    if not G.has_node(subj_label):
                        G.add_node(subj_label, type='UNK', size=12)
                    if not G.has_node(obj_label):
                        G.add_node(obj_label, type='UNK', size=12)

                    rel = verb.lemma_.lower()
                    G.add_edge(subj_label, obj_label, relationship=rel, weight=1)

    return G


def graph_to_pyvis(G, title='KG', bgcolor='#0f0f0f', font_color='white'):
    net = Network(height='700px', width='100%', bgcolor=bgcolor, font_color=font_color)
    net.barnes_hut()
    net.toggle_physics(True)

    type_color = {
        'PERSON':'#ffffff',
        'ORG':'#b3e5fc',
        'GPE':'#a5d6a7',
        'DATE':'#ffe082',
        'NORP':'#ce93d8',
        'PRODUCT':'#ffab91',
        'UNK':'#9e9e9e'
    }

    deg = dict(G.degree())
    for n, attr in G.nodes(data=True):
        ntype = attr.get('type', 'UNK')
        color = type_color.get(ntype, '#9e9e9e')
        size = max(10, int(attr.get('size', 12) + deg.get(n,0)*3))
        net.add_node(n, label=n, title=f"{n} ({ntype})", color=color, size=size)

    for u, v, attr in G.edges(data=True):
        label = attr.get('relationship', '')
        width = min(6, 1 + attr.get('weight', 1))
        net.add_edge(u, v, title=label, value=width)

    return net


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: py_helper.py <input_path> <output_html_path> <orig_name>', file=sys.stderr)
        sys.exit(2)

    input_path = Path(sys.argv[1])
    out_html = Path(sys.argv[2])
    orig_name = sys.argv[3]

    text = read_text_from_path(input_path)
    G = build_graph(text)
    net = graph_to_pyvis(G, title=orig_name)
    html_content = net.generate_html()

    # write file
    out_html.write_text(html_content, encoding='utf8')

    # also print to stdout so the Rust server can return it immediately
    print(html_content)
