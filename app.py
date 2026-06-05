from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from PIL import Image
import pytesseract, io

app = Flask(__name__)
clf = pipeline("text-classification", model="jy46604790/Fake-News-Bert-Detect")

LABELS = {"LABEL_0": "FAKE", "LABEL_1": "REAL"}

def classify(text):
    if not text or not text.strip():
        return {"error": "empty text"}
    out = clf(text[:2000])[0]
    return {"label": LABELS.get(out["label"], out["label"]), "score": round(out["score"], 4)}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")
    file = request.files.get("image")
    if file and file.filename:
        try:
            img = Image.open(io.BytesIO(file.read()))
            text = (text + "\n" + pytesseract.image_to_string(img)).strip()
        except Exception as e:
            return jsonify({"error": f"image: {e}"}), 400
    return jsonify(classify(text) | {"text": text[:500]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
