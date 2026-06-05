from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from PIL import Image
import pytesseract, io, re, sqlite3, time, json, urllib.request, urllib.parse

# ── code-review-graph real imports ────────────────────────────────────────
from crg.graph import GraphStore, NodeInfo, EdgeInfo
from crg.visualization import export_graph_data

app = Flask(__name__)
clf = pipeline("text-classification", model="jy46604790/Fake-News-Bert-Detect")
LABELS = {"LABEL_0": "FAKE", "LABEL_1": "REAL"}

DB_PATH     = os.path.join(os.path.dirname(__file__), "scans.db")
CRG_DB_PATH = os.path.join(os.path.dirname(__file__), "crg_graph.db")

# real GraphStore from code-review-graph repo
graph_store = GraphStore(CRG_DB_PATH)

CREDIBLE   = ["reuters.com","apnews.com","bbc.com","nytimes.com","theguardian.com",
               "washingtonpost.com","bloomberg.com","npr.org","economist.com","wsj.com"]
SUSPICIOUS = ["infowars.com","naturalnews.com","beforeitsnews.com","worldnewsdailyreport.com",
               "empirenews.net","theonion.com","clickhole.com"]
CLICKBAIT  = ["shocking","unbelievable","you won't believe","secret","they don't want you",
              "mainstream media","exposed","wake up","share before deleted","breaking","hoax"]

# ── our scans SQLite (history) ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS scans (
            id TEXT PRIMARY KEY, mode TEXT, label TEXT, score REAL,
            word_count INTEGER, signals TEXT, domain TEXT,
            credibility TEXT, text_snippet TEXT, created_at REAL NOT NULL
        );
        """)

init_db()

# ── analysis ───────────────────────────────────────────────────────────────
def classify(text):
    if not text or not text.strip():
        return {"error": "empty text"}
    word_count = len(text.split())
    cap_ratio  = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclaim    = text.count("!")
    signals    = []
    if word_count < 20:
        return {
            "label": "UNCLEAR", "score": 0.0, "word_count": word_count,
            "signals": ["Too short — BERT needs 20+ words of news-article text"],
            "cap_ratio": round(cap_ratio*100, 1),
            "warning": "Provide a full news headline or article for accurate results."
        }
    out   = clf(text[:2000])[0]
    label = LABELS.get(out["label"], out["label"])
    score = round(out["score"], 4)
    if cap_ratio > 0.15:  signals.append("Excessive capitals")
    if exclaim > 3:       signals.append(f"{exclaim} exclamation marks")
    if word_count < 50:   signals.append("Short text — lower confidence")
    found_cb = [w for w in CLICKBAIT if w.lower() in text.lower()]
    if found_cb: signals.append(f"Clickbait: {', '.join(found_cb[:3])}")
    return {"label": label, "score": score, "word_count": word_count,
            "signals": signals, "cap_ratio": round(cap_ratio*100,1)}

def check_url(url):
    try:
        domain      = urllib.parse.urlparse(url).netloc.replace("www.","")
        credibility = ("credible"   if any(c in domain for c in CREDIBLE)
                  else "suspicious" if any(s in domain for s in SUSPICIOUS)
                  else "unknown")
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            html = r.read().decode("utf-8","ignore")
        text = re.sub(r"\s+"," ", re.sub(r"<[^>]+>","",html)).strip()[:3000]
        result = classify(text)
        result.update({"domain": domain, "credibility": credibility, "text": text[:500]})
        return result
    except Exception as e:
        return {"error": str(e)}

# ── real GraphStore integration ────────────────────────────────────────────
NODE_COLORS = {
    "Input":"#6366f1","Verdict_FAKE":"#ef4444","Verdict_REAL":"#10b981",
    "Verdict_UNCLEAR":"#f59e0b","Model":"#8b5cf6","Score":"#3b82f6",
    "Signal":"#f59e0b","WordCount":"#06b6d4","CapRatio":"#84cc16","Domain":"#ec4899",
}
NODE_ICONS = {
    "Input":"📝","Verdict_FAKE":"🔴","Verdict_REAL":"✅","Verdict_UNCLEAR":"❓",
    "Model":"🧠","Score":"📊","Signal":"⚠️","WordCount":"🔤","CapRatio":"🔠","Domain":"🌐",
}

def build_graph(scan_id, data):
    label   = data.get("label","?")
    score   = data.get("score", 0)
    signals = data.get("signals", [])
    now     = time.time()

    verdict_kind = f"Verdict_{label}"

    # GraphStore generates qualified_name as: file_path::name (or file_path::parent.name)
    # NodeInfo fields: kind, name, file_path, line_start, line_end, language, parent_name, extra
    # EdgeInfo fields: kind, source, target, file_path, line, extra

    def mk_node(kind, name, parent=None, extra=None):
        return NodeInfo(kind=kind, name=name, file_path=scan_id,
                        line_start=0, line_end=0, language="scan",
                        parent_name=parent, extra=extra or {})

    def qn(name, parent=None):
        # mirrors GraphStore._make_qualified
        if parent: return f"{scan_id}::{parent}.{name}"
        return f"{scan_id}::{name}"

    node_infos = [
        mk_node("File", scan_id, extra={"type": "scan"}),   # needed for get_all_files()
        mk_node("Input",      "input",  extra={"mode": data.get("mode","text")}),
        mk_node(verdict_kind, label,    parent="input", extra={"score": score}),
        mk_node("Model",      "BERT",   extra={"model": "jy46604790/Fake-News-Bert-Detect"}),
        mk_node("Score", f"{round(score*100)}%" if label!="UNCLEAR" else "N/A",
                parent="verdict", extra={"raw": score}),
    ]
    if data.get("word_count"):
        node_infos.append(mk_node("WordCount", f"{data['word_count']}words",
                                  extra={"count": data["word_count"]}))
    if data.get("cap_ratio") is not None:
        node_infos.append(mk_node("CapRatio", f"Caps{data['cap_ratio']}pct",
                                  extra={"ratio": data["cap_ratio"]}))
    if data.get("domain"):
        node_infos.append(mk_node("Domain", data["domain"].replace(".","_"),
                                  extra={"credibility": data.get("credibility","unknown")}))
    for i, sig in enumerate(signals):
        node_infos.append(mk_node("Signal", f"signal{i}",
                                  extra={"text": sig[:80]}))

    # Write nodes → GraphStore (upsert_node singular)
    for ni in node_infos:
        graph_store.upsert_node(ni)

    # Build qualified names for edges
    q_input   = qn("input")
    q_model   = qn("BERT")
    q_verdict = qn(label, parent="input")
    q_score   = qn(f"{round(score*100)}%" if label!="UNCLEAR" else "N/A", parent="verdict")

    def mk_edge(kind, src, tgt):
        return EdgeInfo(kind=kind, source=src, target=tgt,
                        file_path=scan_id, line=0, extra={})

    edge_infos = [
        mk_edge("ANALYZED_BY", q_input,   q_model),
        mk_edge("PRODUCES",    q_model,   q_verdict),
        mk_edge("HAS_SCORE",   q_verdict, q_score),
    ]
    if data.get("word_count"):
        edge_infos.append(mk_edge("HAS_STAT", q_input, qn(f"{data['word_count']}words")))
    if data.get("cap_ratio") is not None:
        edge_infos.append(mk_edge("HAS_STAT", q_input, qn(f"Caps{data['cap_ratio']}pct")))
    if data.get("domain"):
        edge_infos.append(mk_edge("FROM_DOMAIN", q_input,
                                  qn(data["domain"].replace(".","_"))))
    for i in range(len(signals)):
        edge_infos.append(mk_edge("HAS_SIGNAL", q_input, qn(f"signal{i}")))

    for ei in edge_infos:
        graph_store.upsert_edge(ei)

    # Export via code-review-graph's export_graph_data, filter to this scan
    raw      = export_graph_data(graph_store)
    scan_qns = set()
    for ni in node_infos:
        if ni.kind == "File":
            scan_qns.add(ni.file_path)
        elif ni.parent_name:
            scan_qns.add(f"{ni.file_path}::{ni.parent_name}.{ni.name}")
        else:
            scan_qns.add(f"{ni.file_path}::{ni.name}")

    nodes_d3, edges_d3 = [], []
    for n in raw.get("nodes", []):
        if n.get("qualified_name","") not in scan_qns:
            continue
        kind  = n.get("kind","")
        vkind = verdict_kind if kind.startswith("Verdict") else kind
        nodes_d3.append({
            "id":    n["qualified_name"],
            "label": n.get("name", kind),
            "kind":  kind,
            "color": NODE_COLORS.get(vkind, NODE_COLORS.get(kind,"#94a3b8")),
            "icon":  NODE_ICONS.get(vkind, NODE_ICONS.get(kind,"•")),
            "meta":  n.get("extra", {}),
        })
    seen = {n["id"] for n in nodes_d3}
    for e in raw.get("edges", []):
        src = e.get("source",""); tgt = e.get("target","")
        if src in seen and tgt in seen:
            edges_d3.append({"source":src,"target":tgt,
                             "kind":e.get("kind",""),"weight":e.get("confidence",1.0)})
    return {"nodes": nodes_d3, "edges": edges_d3}

def save_scan(scan_id, mode, data):
    with get_db() as db:
        db.execute("""INSERT OR REPLACE INTO scans
            (id,mode,label,score,word_count,signals,domain,credibility,text_snippet,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (scan_id, mode, data.get("label"), data.get("score"),
             data.get("word_count"), json.dumps(data.get("signals",[])),
             data.get("domain"), data.get("credibility"),
             data.get("text","")[:300], time.time()))

# ── routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    mode = request.form.get("mode","text")
    data = {}
    if mode == "url":
        url = request.form.get("url","").strip()
        if not url: return jsonify({"error":"no URL"}), 400
        data = check_url(url)
    else:
        text = request.form.get("text","")
        file = request.files.get("image")
        if file and file.filename:
            try:
                img  = Image.open(io.BytesIO(file.read()))
                text = (text+"\n"+pytesseract.image_to_string(img)).strip()
            except Exception as e:
                return jsonify({"error": f"OCR: {e}"}), 400
        data = classify(text)
        data["text"] = text[:500]
    if "error" in data:
        return jsonify(data), 400
    data["mode"] = mode
    scan_id      = f"scan_{int(time.time()*1000)}"
    graph        = build_graph(scan_id, data)
    save_scan(scan_id, mode, data)
    data["graph"]   = graph
    data["scan_id"] = scan_id
    return jsonify(data)

@app.route("/history")
def history():
    with get_db() as db:
        rows = db.execute("""SELECT id,mode,label,score,domain,text_snippet,created_at
                             FROM scans ORDER BY created_at DESC LIMIT 20""").fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/delete/<scan_id>", methods=["DELETE"])
def delete_scan(scan_id):
    with get_db() as db:
        db.execute("DELETE FROM scans WHERE id=?", (scan_id,))
    # remove from GraphStore via raw SQLite
    import sqlite3 as _sq
    conn = _sq.connect(CRG_DB_PATH)
    conn.execute("DELETE FROM nodes WHERE file_path=?", (scan_id,))
    conn.execute("DELETE FROM edges WHERE file_path=?", (scan_id,))
    conn.commit(); conn.close()
    return jsonify({"ok": True})

@app.route("/delete_all", methods=["DELETE"])
def delete_all():
    with get_db() as db:
        db.execute("DELETE FROM scans")
    import sqlite3 as _sq
    conn = _sq.connect(CRG_DB_PATH)
    conn.execute("DELETE FROM nodes"); conn.execute("DELETE FROM edges")
    conn.commit(); conn.close()
    return jsonify({"ok": True})

@app.route("/graph/<scan_id>")
def graph_data(scan_id):
    import sqlite3 as _sq
    conn = _sq.connect(CRG_DB_PATH); conn.row_factory = _sq.Row
    nodes = conn.execute("SELECT * FROM nodes WHERE file_path=?", (scan_id,)).fetchall()
    edges = conn.execute("SELECT * FROM edges WHERE file_path=?", (scan_id,)).fetchall()
    conn.close()
    return jsonify({"nodes":[dict(n) for n in nodes],"edges":[dict(e) for e in edges]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
