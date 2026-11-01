
import os
import threading

import numpy as np
from flask import Flask, jsonify, render_template, request

import expert_system as xs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
MAX_SYMPTOMS = 40

app = Flask(__name__)


es = None
LOAD_ERROR = None
try:
    es = xs.load_expert_system(DATA_DIR)
    VALID = set(es.symptoms)
except FileNotFoundError as exc:          
    LOAD_ERROR = (
        "Training.csv was not found. "
    )
    VALID = set()
    print("ERROR:", exc)


def label(sym):
    return xs.pretty(sym)


def serialize(d):
    return {
        "disease": d.disease,
        "certainty": round(d.certainty * 100, 1),
        "symptom_certainty": round(d.base_certainty * 100, 1),
        "matched": [label(s) for s in d.matched],
        "missing": [label(s) for s in d.missing[:8]],
        "history_notes": d.history_notes,
        "rule": es.rule_text(d.disease),
        "description": es.descriptions.get(d.disease),
        "precautions": es.precautions.get(d.disease, []),
    }


def need_system():
    if es is None:
        return jsonify(error=LOAD_ERROR), 503
    return None


@app.get("/")
def index():
    return render_template("index.html", load_error=LOAD_ERROR)


@app.get("/api/symptoms")
def api_symptoms():
    err = need_system()
    if err:
        return err
    return jsonify(symptoms=[{"id": s, "label": label(s)} for s in sorted(es.symptoms)])


@app.get("/api/history")
def api_history():
    return jsonify(history=[{"id": k, "label": v[0]} for k, v in xs.HISTORY_RULES.items()])


@app.post("/api/diagnose")
def api_diagnose():
    err = need_system()
    if err:
        return err
    data = request.get_json(silent=True) or {}

    raw = data.get("symptoms", [])
    if not isinstance(raw, list):
        return jsonify(error="symptoms must be a list"), 400
    symptoms = []
    for s in raw[:MAX_SYMPTOMS]:
        if isinstance(s, str) and s in VALID and s not in symptoms:
            symptoms.append(s)
    if not symptoms:
        return jsonify(error="Please choose at least one valid symptom."), 400

    age = data.get("age")
    try:
        age = int(age) if age not in (None, "") else None
        if age is not None and not 0 < age < 120:
            age = None
    except (TypeError, ValueError):
        age = None

    history = [h for h in data.get("history", []) if h in xs.HISTORY_RULES]

    ranked = es.diagnose(symptoms, age, history)
    follow = es.follow_up_questions(ranked, set(symptoms)) if ranked else []
    return jsonify(
        symptoms_used=[label(s) for s in symptoms],
        results=[serialize(d) for d in ranked[:5]],
        followups=[{"id": s, "label": label(s)} for s in follow],
    )


_metrics_cache = {}
_metrics_lock = threading.Lock()


@app.get("/api/metrics")
def api_metrics():
    
    err = need_system()
    if err:
        return err
    with _metrics_lock:
        if not _metrics_cache:
            df, lab = es._df, es._label_col
            rng = np.random.RandomState(42)
            test_idx = []
            for _, idx in df.groupby(lab).groups.items():
                idx = np.array(list(idx))
                rng.shuffle(idx)
                test_idx.extend(idx[: max(1, int(0.2 * len(idx)))])
            test, train = df.loc[test_idx], df.drop(index=test_idx)
            if len(test) > xs.EVAL_LIMIT:          # keep big datasets (DDXPlus) fast
                test = test.sample(n=xs.EVAL_LIMIT, random_state=0)
            model = xs.ExpertSystem(es.min_freq, es.min_symptoms, es.severity)
            model.build(train, lab, es.symptoms)
            top1 = top3 = 0
            for row, truth in zip(test[es.symptoms].values, test[lab].values):
                syms = [s for s, v in zip(es.symptoms, row) if v == 1]
                names = [d.disease for d in model.diagnose(syms)]
                top1 += bool(names and names[0] == truth)
                top3 += truth in names[:3]
            n = len(test)
            _metrics_cache.update(
                diseases=len(es.rules), symptoms=len(es.symptoms), test_cases=n,
                top1=round(100 * top1 / n, 1), top3=round(100 * top3 / n, 1))
    return jsonify(_metrics_cache)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
