"""Customer Segmentation API.

Standard library + Flask only (no pandas / scikit-learn needed to run it).

Endpoints
  GET  /                  health check
  GET  /segments          all customers   (?segment=Champions & limit=100)
  GET  /summary           per-segment size and average RFM
  GET  /top-customers     top spenders    (?n=20)
  GET  /metrics           model quality metrics from training
  GET|POST /predict       segment for a NEW customer (recency, frequency, monetary)
"""
import os
import csv
import json
import math
from statistics import mean

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, "ml")
SEGMENTS_FILE = os.path.join(ML_DIR, "customer_segments.csv")
METRICS_FILE = os.path.join(ML_DIR, "metrics.json")
MODEL_FILE = os.path.join(ML_DIR, "model.json")

_cache = {}


def _cached(path, loader):
    """Reload a file only when it changed on disk (so retraining is picked up)."""
    mtime = os.path.getmtime(path)
    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    data = loader(path)
    _cache[path] = (mtime, data)
    return data


def _load_segments(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "CustomerID": int(r["CustomerID"]),
                "Recency": int(float(r["Recency"])),
                "Frequency": float(r["Frequency"]),
                "Monetary": float(r["Monetary"]),
                "Avg_Order_Value": float(r["Avg_Order_Value"]),
                "Cluster": int(r["Cluster"]),
                "Segment": r["Segment"],
            })
    return rows


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def segments_data():
    return _cached(SEGMENTS_FILE, _load_segments)


def _int_arg(name, default, lo=1, hi=100000):
    try:
        return max(lo, min(hi, int(request.args.get(name, default))))
    except (TypeError, ValueError):
        return default


@app.errorhandler(FileNotFoundError)
def missing_file(e):
    return jsonify(error="Model files not found. Run 'python train_model.py' first."), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify(error="Not found"), 404


@app.route("/")
def home():
    return jsonify(status="running", project="Customer Segmentation")


@app.route("/segments")
def segments():
    rows = segments_data()
    seg = request.args.get("segment")
    if seg:
        rows = [r for r in rows if r["Segment"].lower() == seg.lower()]
    if "limit" in request.args:
        rows = rows[:_int_arg("limit", len(rows))]
    return jsonify(rows)


@app.route("/summary")
def summary():
    groups = {}
    for r in segments_data():
        groups.setdefault(r["Segment"], []).append(r)
    out = []
    for name, rows in groups.items():
        out.append({
            "Segment": name,
            "Customers": len(rows),
            "Avg_Recency_days": round(mean(r["Recency"] for r in rows), 1),
            "Avg_Frequency": round(mean(math.expm1(r["Frequency"]) for r in rows), 2),
            "Avg_Monetary": round(mean(math.expm1(r["Monetary"]) for r in rows), 2),
        })
    out.sort(key=lambda d: d["Avg_Monetary"], reverse=True)
    return jsonify(out)


@app.route("/top-customers")
def top_customers():
    n = _int_arg("n", 20, hi=1000)
    top = sorted(segments_data(), key=lambda c: c["Monetary"], reverse=True)[:n]
    return jsonify(top)


@app.route("/metrics")
def metrics():
    return jsonify(_cached(METRICS_FILE, _load_json))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Assign a segment to a new customer using the trained KMeans centroids.

    Send raw values: recency (days since last purchase), frequency (number of
    purchases), monetary (total spend). Example:
      /predict?recency=30&frequency=12&monetary=45000
    """
    params = request.get_json(silent=True) or request.args
    try:
        raw = [float(params[k]) for k in ("recency", "frequency", "monetary")]
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Provide numeric recency, frequency and monetary."), 400
    if any(v < 0 for v in raw):
        return jsonify(error="Values must be zero or positive."), 400

    model = _cached(MODEL_FILE, _load_json)
    z = [(math.log1p(v) - m) / s
         for v, m, s in zip(raw, model["scaler_mean"], model["scaler_scale"])]
    dists = [math.dist(z, c) for c in model["centroids"]]
    cluster = dists.index(min(dists))
    return jsonify(
        cluster=cluster,
        segment=model["segment_names"][str(cluster)],
        input=dict(recency=raw[0], frequency=raw[1], monetary=raw[2]),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
