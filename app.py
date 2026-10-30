"""
JalDrushti — Flask Application
Drought Severity Classifier using Real IMD Rainfall Data
"""
from flask import Flask, render_template, request, jsonify
import joblib, json, numpy as np

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)

model  = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
le     = joblib.load("models/label_encoder.pkl")

with open("models/metrics.json") as f:
    metrics = json.load(f)

SEVERITY_CONFIG = {
    "Severe Drought":   {"color": "#ef4444", "icon": "🆘", "bg": "rgba(239,68,68,0.1)",
                         "action": "Emergency protocol. Dispatch tankers immediately."},
    "Moderate Drought": {"color": "#f97316", "icon": "🚨", "bg": "rgba(249,115,22,0.1)",
                         "action": "Alert district office. Restrict water usage."},
    "Normal":           {"color": "#22c55e", "icon": "💧", "bg": "rgba(34,197,94,0.1)",
                         "action": "No immediate action needed."},
    "Excess Rainfall":  {"color": "#3b82f6", "icon": "🌊", "bg": "rgba(59,130,246,0.1)",
                         "action": "Monitor flood risk. Check drainage systems."},
}

MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]

@app.route("/")
def index():
    return render_template("index.html", metrics=metrics,
                           severity_config=SEVERITY_CONFIG)

@app.route("/predict", methods=["POST"])
def predict():
    data   = request.json
    months = [float(data.get(m, 0)) for m in MONTHS]

    monsoon     = sum(months[5:9])           # Jun–Sep
    non_monsoon = sum(months) - monsoon
    annual      = sum(months)

    # Approximate subdivision mean monsoon (avg from dataset ~470mm)
    monsoon_deviation = ((monsoon - 470) / 470) * 100

    features = np.array([months + [monsoon, non_monsoon, monsoon_deviation]])
    scaled   = scaler.transform(features)
    pred_enc = model.predict(scaled)[0]
    label    = le.inverse_transform([pred_enc])[0]
    proba    = model.predict_proba(scaled)[0]

    classes_with_prob = [
        {"label": le.classes_[i], "prob": round(float(p) * 100, 1)}
        for i, p in enumerate(proba)
    ]
    classes_with_prob.sort(key=lambda x: x["prob"], reverse=True)

    config = SEVERITY_CONFIG[label]
    return jsonify({
        "severity":    label,
        "probabilities": classes_with_prob,
        "monsoon_mm":  round(monsoon, 1),
        "deviation":   round(monsoon_deviation, 1),
        **config
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
