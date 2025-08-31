![alt text](image.png)



# Flask Knowledge Graph App

This project is a lightweight **web application** that lets you upload a text file and see an **interactive knowledge graph** built from its content.  
It uses **Flask** for the web interface, **NetworkX** for the graph, and **PyVis** for visualization.

---

## Features
- Upload `.txt` documents through a local webpage.
- Automatic file encoding detection (`chardet`) → avoids Unicode errors.
- Simple graph construction using co-occurrence of entities.
- Interactive visualization in your browser (zoom, drag, highlight).

---

## Requirements
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended)  

---

## Installation

Clone your project and install dependencies:

```bash
uv add flask pyvis chardet networkx
````

---

## Running the App

Start the Flask server:

```bash
uv run flask --app app run --debug
```

Open your browser at:

```
http://127.0.0.1:5000
```

---

## Usage

1. Upload a `.txt` file when prompted.
2. The server will:

   * Detect encoding automatically,
   * Read the file safely,
   * Build a knowledge graph,
   * Render an interactive graph in the browser.
3. No files are permanently saved — HTML is generated in memory.

---

## Example

Upload a file like:

```
Alice went to Paris. Bob met Alice in London. Charlie knows Bob.
```

The resulting graph will contain:

* Nodes: `Alice`, `Bob`, `Charlie`, `Paris`, `London`
* Edges: showing co-occurrence relationships.

---

## License

MIT

```
---
```
