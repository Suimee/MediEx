# MediEx - Medical Diagnosis Expert System (website)

Flask website around a rule-based expert system (forward chaining, certainty factors,
explanations, follow-up questions). **Educational use only - not medical advice.**

## Project layout

```
mediexpert/
  app.py               Flask server + JSON API
  expert_system.py     the expert system (knowledge base, inference, explanation)
  requirements.txt
  data/                <- put the Kaggle CSV files here
  templates/index.html
  static/style.css
  static/app.js
```

## Run on your computer

1. Install Python 3.9 or newer (https://python.org). Tick "Add Python to PATH" on Windows.
2. Copy the Kaggle CSV files into the `data` folder (see `data/README.txt`).
3. Open a terminal in the project folder and run:

**Windows (PowerShell)**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

**macOS / Linux**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

4. Open http://127.0.0.1:5000 in your browser.

If the page shows "Training.csv was not found", the CSV files are not in `data/`.
To keep the CSVs elsewhere: `set DATA_DIR=C:\path\to\csvs` (Windows) or
`export DATA_DIR=/path/to/csvs` (macOS/Linux) before `python app.py`.

## Put it online (optional)

Kaggle notebooks cannot host a public website, so deploy to a Python host
such as **Render** or **PythonAnywhere**:

1. Upload the project to a GitHub repository (include the `data` folder, license permitting).
2. Create a new "Web Service" from that repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`

(On PythonAnywhere, use their Flask setup and point the WSGI file to `app.app`.)

## How the pieces connect

- `expert_system.py` builds the rule base from `Training.csv` when the server starts.
- `GET /api/symptoms`  list of known symptoms (powers the search box)
- `POST /api/diagnose` body: `{"symptoms": ["high_fever", ...], "age": 30, "history": ["diabetes"]}`
  returns ranked diseases, explanations and follow-up questions
- `GET /api/metrics`   hold-out accuracy shown in the About section

## Ideas for extra marks

- Add more history rules in `HISTORY_RULES` (expert_system.py) with medical references.
- Tune `min_freq` / `min_symptoms` in `load_expert_system()` and report the effect.
- Compare with a Naive Bayes baseline in your report.
