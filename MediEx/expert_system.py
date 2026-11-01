
import argparse
import difflib
import math
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


TRAIN_NAMES = ["Training.csv", "Train.csv", "dataset.csv",
               "release_validate_patients.csv", "release_train_patients.csv",
               "validate.csv", "train.csv",
               "release_validate_patients", "release_train_patients"]   # no extension
TEST_NAMES = ["Testing.csv", "Test.csv", "release_test_patients.csv", "test.csv",
              "release_test_patients"]
DDX_TRAIN_ROWS = 50000     
EVAL_LIMIT = 3000          



HISTORY_RULES = {
    "diabetes": ("Known diabetes",
                 ["diabetes", "hypoglycemia", "urinary tract infection"], 0.10),
    "hypertension": ("Known high blood pressure",
                     ["hypertension", "heart attack", "paralysis"], 0.10),
    "asthma": ("Known asthma / chronic lung disease",
               ["bronchial asthma", "pneumonia", "tuberculosis"], 0.10),
    "smoker": ("Smoker",
               ["tuberculosis", "pneumonia", "bronchial asthma", "heart attack"], 0.08),
    "mosquito_area": ("Lives in / travelled to mosquito-borne disease area",
                      ["dengue", "malaria"], 0.12),
    "unsafe_water": ("Recently drank unsafe water / ate street food",
                     ["typhoid", "hepatitis a", "gastroenteritis", "jaundice"], 0.10),
    "alcohol": ("Regular alcohol use",
                ["alcoholic hepatitis", "gerd", "peptic ulcer"], 0.10),
    "family_heart": ("Family history of heart disease",
                     ["heart attack", "hypertension"], 0.08),
}

# Age-based rules: (label, predicate, [disease fragments], boost)
AGE_RULES = [
    ("Age 50 or above", lambda a: a >= 50,
     ["heart attack", "hypertension", "diabetes", "osteoarthritis",
      "cervical spondylosis", "paralysis"], 0.05),
    ("Child (age 12 or below)", lambda a: a <= 12,
     ["chicken pox", "common cold"], 0.05),
]

def norm(name):
    """Normalise symptom names: 'dischromic _patches' -> 'dischromic_patches'."""
    s = re.sub(r"\s+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", s)


def pretty(sym):
    return sym.replace("_", " ")


def find_file(data_dir, candidates):
 
    try:
        existing = {f.lower(): f for f in os.listdir(data_dir)}
    except FileNotFoundError:
        return None
    for c in candidates:
        if c.lower() in existing:
            return os.path.join(data_dir, existing[c.lower()])
    return None


def _from_list_format(raw):
    label = [c for c in raw.columns if str(c).strip().lower() == "disease"][0]
    sym_cols = [c for c in raw.columns if c != label and not str(c).startswith("Unnamed")]
    long = raw.melt(id_vars=[label], value_vars=sym_cols, ignore_index=False)
    long = long.dropna(subset=["value"])
    long["value"] = long["value"].map(norm)
    long = long[long["value"] != ""]
    onehot = pd.crosstab(long.index, long["value"]).clip(upper=1)
    df = onehot.reindex(raw.index, fill_value=0).astype("int8")
    df["prognosis"] = raw[label].astype(str).str.strip().values
    return df


def _load_json(folder, name):
    path = find_file(folder, [name])
    if not path:
        return {}
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _from_ddxplus(path, max_rows):
    raw = pd.read_csv(path, nrows=max_rows or DDX_TRAIN_ROWS)
    folder = os.path.dirname(os.path.abspath(path))
    ev_json = _load_json(folder, "release_evidences.json")
    cond_json = _load_json(folder, "release_conditions.json")
    if isinstance(ev_json, list):
        ev_json = {e.get("name"): e for e in ev_json if isinstance(e, dict)}
    if isinstance(cond_json, list):
        cond_json = {c.get("condition_name"): c for c in cond_json if isinstance(c, dict)}

    labels, used = {}, set()

    def label_for(token):
        if token in labels:
            return labels[token]
        base, _, value = token.partition("_@_")
        info = ev_json.get(base, {})
        text = (info.get("question_en") or base).strip()
        if value:
            meaning = info.get("value_meaning", {}).get(value, {})
            text += " - " + str(meaning.get("en") or value)
        text = norm(text)
        if text in used:                      # keep names unique
            text = f"{text}_[{token}]"
        used.add(text)
        labels[token] = text
        return text

    rows, cols = [], []
    for i, ev in enumerate(raw["EVIDENCES"].astype(str).values):
        for tok in set(re.findall(r"[^\s,'\"\[\]]+", ev)):
            rows.append(i)
            cols.append(label_for(tok))
    long = pd.DataFrame({"row": rows, "ev": cols})
    onehot = pd.crosstab(long["row"], long["ev"]).clip(upper=1)
    df = onehot.reindex(range(len(raw)), fill_value=0).astype("int8")
    df.index = raw.index

    def disease_name(x):
        x = str(x).strip()
        return str(cond_json.get(x, {}).get("cond-name-eng") or x).strip()

    df["prognosis"] = raw["PATHOLOGY"].map(disease_name).values
    return df


def read_symptom_table(path, max_rows=None):
    head = pd.read_csv(path, nrows=0)
    lower = [str(c).strip().lower() for c in head.columns]
    if "evidences" in lower and "pathology" in lower:
        df = _from_ddxplus(path, max_rows)          # DDXPlus patient file
    else:
        df = pd.read_csv(path)
    if "disease" in lower and any(c.startswith("symptom") for c in lower):
        df = _from_list_format(df)          # itachi9604-style dataset.csv
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [norm(c) if c.strip().lower() != "prognosis" else "prognosis"
                  for c in df.columns]
    label_col = "prognosis" if "prognosis" in df.columns else df.columns[-1]
    df[label_col] = df[label_col].astype(str).str.strip()
    sym_cols = [c for c in df.columns if c != label_col]
    df[sym_cols] = df[sym_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("int8")
    return df, label_col, sym_cols


@dataclass
class Diagnosis:
    disease: str
    certainty: float
    base_certainty: float
    matched: list
    missing: list
    history_notes: list = field(default_factory=list)


class ExpertSystem:
    def __init__(self, min_freq=0.2, min_symptoms=4, severity=None,
                 descriptions=None, precautions=None):
        self.min_freq = min_freq
        self.min_symptoms = min_symptoms
        self.severity = severity or {}
        self.descriptions = descriptions or {}
        self.precautions = precautions or {}
        self.rules = {}     
        self.weight = {}    
        self.symptoms = []


    def build(self, df, label_col, sym_cols):
   
        self.symptoms = list(sym_cols)
        self.rules = {}
        for disease, g in df.groupby(label_col):
            freq = g[sym_cols].mean()
            keep = freq[freq >= self.min_freq]
            if len(keep) < self.min_symptoms:           # make sure rule is not tiny
                keep = freq[freq > 0].nlargest(self.min_symptoms)
            self.rules[disease] = {s: float(f) for s, f in keep.items()}

        n_dis = max(len(self.rules), 1)
        for s in self.symptoms:
            n_with = sum(1 for r in self.rules.values() if s in r)
            specificity = math.log(1 + n_dis / max(n_with, 1))
            self.weight[s] = float(self.severity.get(s, 1)) * specificity
        return self

    def diagnose(self, symptoms, age=None, history=None):
        symptoms = {norm(s) for s in symptoms}
        history = set(history or [])
        user_total = sum(self.weight.get(s, 1.0) for s in symptoms) or 1.0
        results = []

        for disease, rule in self.rules.items():
            matched = sorted(symptoms & rule.keys())
            if not matched:                 
                continue
            rule_total = sum(self.weight[s] for s in rule)
            matched_w = sum(self.weight[s] for s in matched)
            coverage = matched_w / rule_total           
            relevance = matched_w / user_total          
            base = 0.5 * coverage + 0.5 * relevance

            boost, notes = 0.0, []
            dl = disease.lower()
            for key in history:
                if key in HISTORY_RULES:
                    label, frags, b = HISTORY_RULES[key]
                    if any(f in dl for f in frags):
                        boost += b
                        notes.append(f"{label} (+{b:.2f})")
            if age is not None:
                for label, pred, frags, b in AGE_RULES:
                    if pred(age) and any(f in dl for f in frags):
                        boost += b
                        notes.append(f"{label} (+{b:.2f})")

            cert = min(1.0, base + boost)
            missing = sorted((s for s in rule if s not in symptoms),
                             key=lambda s: -self.weight[s])
            results.append(Diagnosis(disease, cert, base, matched, missing, notes))

        results.sort(key=lambda d: d.certainty, reverse=True)
        return results

    # ---------------- backward-chaining style follow-up ----------------
    def follow_up_questions(self, ranked, asked, top_n=3, max_q=5):
        """Symptoms that would best confirm/reject the top hypotheses."""
        scores = {}
        for d in ranked[:top_n]:
            for s in d.missing:
                if s not in asked:
                    scores[s] = scores.get(s, 0.0) + self.weight[s] * d.certainty
        return [s for s, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:max_q]

    # ---------------- explanation facility ----------------
    def rule_text(self, disease, max_syms=8):
        rule = self.rules[disease]
        top = sorted(rule, key=lambda s: -self.weight[s])[:max_syms]
        more = len(rule) - len(top)
        body = ", ".join(pretty(s) for s in top) + (f", +{more} more" if more > 0 else "")
        return f"IF patient shows {{{body}}} THEN {disease}  (certainty grows with matched symptoms)"

    def explain(self, d):
        lines = [f"  Rule fired : {self.rule_text(d.disease)}",
                 f"  Matched    : {', '.join(pretty(s) for s in d.matched)}"]
        if d.missing:
            lines.append("  Not reported: " + ", ".join(pretty(s) for s in d.missing[:6]))
        lines.append(f"  Symptom certainty: {d.base_certainty:.2f}")
        if d.history_notes:
            lines.append("  History/age adjustments: " + "; ".join(d.history_notes))
        lines.append(f"  Final certainty  : {d.certainty:.2f}")
        return "\n".join(lines)

    def export_rules(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("KNOWLEDGE BASE (auto-derived from data)\n")
            f.write(f"min_freq={self.min_freq}, min_symptoms={self.min_symptoms}\n\n")
            for disease in sorted(self.rules):
                f.write(f"RULE: {disease}\n")
                for s, fr in sorted(self.rules[disease].items(), key=lambda kv: -kv[1]):
                    f.write(f"    {pretty(s):35s} freq={fr:.2f}  weight={self.weight[s]:.2f}\n")
                f.write("\n")


def evaluate(es, df, label_col, sym_cols, title):
    top1 = top3 = 0
    if len(df) > EVAL_LIMIT:                       
        df = df.sample(n=EVAL_LIMIT, random_state=0)
    X = df[sym_cols].values
    y = df[label_col].values
    for row, truth in zip(X, y):
        syms = [s for s, v in zip(sym_cols, row) if v == 1]
        ranked = es.diagnose(syms)
        names = [d.disease for d in ranked]
        top1 += bool(names and names[0] == truth)
        top3 += truth in names[:3]
    n = len(df)
    print(f"{title}: {n} cases | Top-1 accuracy = {top1 / n:.3f} | Top-3 accuracy = {top3 / n:.3f}")
    return top1 / n, top3 / n


def run_eval(args, df, label_col, sym_cols, severity, descs, precs, data_dir):
    # 1) Hold-out split of the training data (stratified, 80/20)
    rng = np.random.RandomState(42)
    test_idx = []
    for _, idx in df.groupby(label_col).groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        test_idx.extend(idx[: max(1, int(0.2 * len(idx)))])
    test = df.loc[test_idx]
    train = df.drop(index=test_idx)
    es = ExpertSystem(args.min_freq, args.min_symptoms, severity, descs, precs)
    es.build(train, label_col, sym_cols)
    evaluate(es, test, label_col, sym_cols, "Hold-out (20% of the training file)")

    # 2) Official Testing.csv, knowledge base built from the full Training.csv
    tpath = find_file(data_dir, TEST_NAMES)
    if tpath and os.path.abspath(tpath) != os.path.abspath(find_file(data_dir, TRAIN_NAMES) or ""):
        tdf, tlabel, tcols = read_symptom_table(tpath, max_rows=EVAL_LIMIT)
        for c in sym_cols:                      # align columns with training
            if c not in tdf.columns:
                tdf[c] = 0
        es_full = ExpertSystem(args.min_freq, args.min_symptoms, severity, descs, precs)
        es_full.build(df, label_col, sym_cols)
        evaluate(es_full, tdf, tlabel, sym_cols, "Separate test file")
    else:
        print("No separate test file found - skipped.")
    print("\nNOTE: this dataset is largely synthetic with many near-duplicate rows, "
          "so high accuracy does not prove real clinical performance.")


# --------------------------------------------------------------------------
# Interactive interface
# --------------------------------------------------------------------------
def yes_no(prompt):
    while True:
        a = input(prompt + " (y/n): ").strip().lower()
        if a in ("y", "yes"):
            return True
        if a in ("n", "no", ""):
            return False


def match_symptom(text, es):
    """Map free text to a known symptom (exact -> substring -> fuzzy)."""
    key = norm(text)
    if key in es.symptoms:
        return key
    subs = [s for s in es.symptoms if key and key in s]
    cands = subs[:5] if subs else difflib.get_close_matches(key, es.symptoms, n=5, cutoff=0.6)
    if not cands:
        print(f"  ! '{text}' not recognised.")
        return None
    if len(cands) == 1:
        return cands[0] if yes_no(f"  Did you mean '{pretty(cands[0])}'?") else None
    print(f"  '{text}' could be:")
    for i, c in enumerate(cands, 1):
        print(f"    {i}. {pretty(c)}")
    pick = input("  Choose number (Enter to skip): ").strip()
    return cands[int(pick) - 1] if pick.isdigit() and 1 <= int(pick) <= len(cands) else None


def collect_symptoms(es):
    print("\nEnter symptoms separated by commas (e.g. high fever, headache, vomiting).")
    print("Type 'list' to see all known symptoms.")
    while True:
        raw = input("Symptoms: ").strip()
        if raw.lower() == "list":
            print(", ".join(pretty(s) for s in es.symptoms))
            continue
        found = []
        for part in filter(None, (p.strip() for p in raw.split(","))):
            m = match_symptom(part, es)
            if m and m not in found:
                found.append(m)
        if found:
            return found
        print("No valid symptoms entered, try again.")


def collect_history():
    age = None
    while age is None:
        a = input("Age (Enter to skip): ").strip()
        if not a:
            break
        if a.isdigit() and 0 < int(a) < 120:
            age = int(a)
    history = [k for k, (label, _, _) in HISTORY_RULES.items() if yes_no(f"{label}?")]
    return age, history


def show_results(es, ranked, top_k=5):
    if not ranked:
        print("\nNo disease rule matched these symptoms.")
        return
    print("\n=== POSSIBLE DIAGNOSES (ranked by certainty) ===")
    for i, d in enumerate(ranked[:top_k], 1):
        print(f"{i}. {d.disease:35s} certainty = {d.certainty * 100:5.1f}%")
    print("\n=== EXPLANATION (why?) ===")
    for d in ranked[:3]:
        print(f"\n{d.disease}\n{es.explain(d)}")
    best = ranked[0].disease
    if best in es.descriptions:
        print(f"\nAbout {best}: {es.descriptions[best]}")
    if best in es.precautions:
        print(f"Suggested precautions for {best}:")
        for p in es.precautions[best]:
            print(f"  - {p}")
    


def consult(es, symptoms=None, age=None, history=None):
    
    interactive = symptoms is None
    if interactive:
        symptoms = collect_symptoms(es)
        age, history = collect_history()
    symptoms = [norm(s) for s in symptoms]
    ranked = es.diagnose(symptoms, age, history)
    show_results(es, ranked)
    if interactive and ranked:
        asked = set(symptoms)
        if yes_no("Answer a few follow-up questions to refine the result?"):
            for s in es.follow_up_questions(ranked, asked):
                asked.add(s)
                if yes_no(f"Do you have {pretty(s)}?"):
                    symptoms.append(s)
            print("\n--- Refined result ---")
            show_results(es, es.diagnose(symptoms, age, history))


# --------------------------------------------------------------------------
# Data loading / main
# --------------------------------------------------------------------------
def load_optional(data_dir):
    severity, descs, precs = {}, {}, {}
    p = find_file(data_dir, ["Symptom-severity.csv", "Symptom_severity.csv", "symptom_severity.csv"])
    if p:
        sv = pd.read_csv(p)
        # works for both Kaggle versions (column names differ): col 0 = symptom, col 1 = weight
        severity = {norm(r.iloc[0]): float(r.iloc[1]) for _, r in sv.iterrows()
                    if pd.notna(r.iloc[1])}
    p = find_file(data_dir, ["symptom_Description.csv", "disease_description.csv"])
    if p:
        dd = pd.read_csv(p)
        dd.columns = [c.strip().lower() for c in dd.columns]
        descs = {str(r.iloc[0]).strip(): str(r.iloc[1]).strip() for _, r in dd.iterrows()}
    p = find_file(data_dir, ["symptom_precaution.csv", "disease_precaution.csv"])
    if p:
        pr = pd.read_csv(p)
        precs = {str(r.iloc[0]).strip(): [str(x).strip() for x in r.iloc[1:] if pd.notna(x)]
                 for _, r in pr.iterrows()}
    return severity, descs, precs


def find_data_dir(data_dir=None):
 
    roots = [data_dir] if data_dir else []
    roots += ["/kaggle/input", ".", os.getcwd()]
    for root in roots:
        if root and find_file(root, TRAIN_NAMES):
            return root
    for root in roots:
        if root and os.path.isdir(root):
            for dirpath, _, files in os.walk(root):
                if any(f.lower() in {n.lower() for n in TRAIN_NAMES} for f in files):
                    return dirpath
    return None


def load_expert_system(data_dir=None, min_freq=0.2, min_symptoms=4):

    d = find_data_dir(data_dir)
    if d is None:
        raise FileNotFoundError(
            "Training.csv not found.")
    df, label_col, sym_cols = read_symptom_table(find_file(d, TRAIN_NAMES))
    severity, descs, precs = load_optional(d)
    es = ExpertSystem(min_freq, min_symptoms, severity, descs, precs)
    es.build(df, label_col, sym_cols)
    es.data_dir, es._df, es._label_col = d, df, label_col
    print(f"Data folder: {d}")
    print(f"Knowledge base ready: {len(es.rules)} diseases, {len(es.symptoms)} symptoms.")
    return es


def resolve_symptoms(es, text):
    
    found, unknown = [], []
    for part in filter(None, (p.strip() for p in re.split(r"[,;\n]", text))):
        key = norm(part)
        if key in es.symptoms:
            hit = key
        else:
            subs = [s for s in es.symptoms if key in s]
            pool = subs or es.symptoms
            close = difflib.get_close_matches(key, pool, n=1, cutoff=0.0 if subs else 0.7)
            hit = close[0] if close else None
        if hit is None:
            unknown.append(part)
            continue
        if hit != key:
            print(f"  (understood '{part}' as '{pretty(hit)}')")
        if hit not in found:
            found.append(hit)
    return found, unknown


def predict(es, symptoms_text, age=None, history=None, top_k=5):
  
    syms, unknown = resolve_symptoms(es, symptoms_text)
    if unknown:
        print("Not recognised (use list_symptoms(es) to see valid names):", ", ".join(unknown))
    if not syms:
        print("No valid symptoms given.")
        return []
    print("Symptoms used:", ", ".join(pretty(s) for s in syms))
    ranked = es.diagnose(syms, age, history)
    show_results(es, ranked, top_k)
    return ranked


def list_symptoms(es):
    print(", ".join(pretty(s) for s in es.symptoms))


def main():
    ap = argparse.ArgumentParser(description="Medical Diagnosis Expert System")
    ap.add_argument("--data-dir", default=".", help="folder containing the CSV files")
    ap.add_argument("--min-freq", type=float, default=0.2,
                    help="keep a symptom in a disease rule if it appears in >= this fraction of that disease's rows")
    ap.add_argument("--min-symptoms", type=int, default=4, help="minimum symptoms per rule")
    ap.add_argument("--eval", action="store_true", help="evaluate accuracy and exit")
    ap.add_argument("--export-rules", metavar="FILE", help="write the knowledge base to FILE and exit")
    ap.add_argument("--symptoms", help="comma-separated symptoms (non-interactive demo)")
    ap.add_argument("--age", type=int)
    ap.add_argument("--history", help="comma-separated keys: " + ", ".join(HISTORY_RULES))
    args, _ = ap.parse_known_args()      # ignores extra notebook/kernel arguments

    found_dir = find_data_dir(None if args.data_dir == "." else args.data_dir) \
        or find_data_dir(".")
    if not found_dir:
        sys.exit(f"Training.csv not found in '{args.data_dir}'. "
                 f"and use --data-dir <folder>.")
    args.data_dir = found_dir
    train_path = find_file(args.data_dir, TRAIN_NAMES)
    df, label_col, sym_cols = read_symptom_table(train_path)
    severity, descs, precs = load_optional(args.data_dir)

    if args.eval:
        run_eval(args, df, label_col, sym_cols, severity, descs, precs, args.data_dir)
        return

    es = ExpertSystem(args.min_freq, args.min_symptoms, severity, descs, precs)
    es.build(df, label_col, sym_cols)
    print(f"Knowledge base ready: {len(es.rules)} diseases, {len(es.symptoms)} symptoms.")

    if args.export_rules:
        es.export_rules(args.export_rules)
        print(f"Rules written to {args.export_rules}")
        return

    if args.symptoms:
        syms = [s.strip() for s in args.symptoms.split(",") if s.strip()]
        hist = [h.strip() for h in args.history.split(",")] if args.history else []
        consult(es, syms, args.age, hist)
        return

    while True:
        consult(es)
        if not yes_no("Run another consultation?"):
            break


if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()         
