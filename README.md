# Bayesian Network Lab — DNA Case

A small Streamlit app that demonstrates a simple Bayesian network for a DNA forensic case. It lets you explore how the prior probability and DNA lab error affect the posterior probability of guilt.

It follows R v Adams (1996).

**Quick links**
- App: `BN.py`
- Tests: `tests/`
- CI: `.github/workflows/ci.yml`

**Requirements**
- Python 3.12+
- See `requirements.txt` for runtime dependencies.

**Setup (local)**
1. Create and activate a virtual environment:

   - macOS / Linux:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   - Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
# pytest is installed in CI; install locally if you plan to run tests:
pip install pytest
```

**Run the app**
```bash
streamlit run BN.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

**What to try**
- Use the **Lab Error Rate** slider (sidebar) to add false-positive probability to the DNA CPT.
- Use the **Prior P(Guilty)** slider (sidebar) to change the prior; the Guilty CPT updates immediately unless you manually edited it in the CPT editor.
- Use the CPT editor (Advanced) to inspect or override CPDs — note that manual edits prevent the sliders from overwriting that CPT until you reset it.

**Tests**
Run the test suite locally:
```bash
pytest -q
```

**CI**
There is a GitHub Actions workflow at `.github/workflows/ci.yml` that runs the tests on push and pull requests to `main`. The workflow installs `requirements.txt` (if present) and `pytest` and fails the check if tests fail.

**Notes**
- The app uses `pgmpy` for Bayes network modeling and `streamlit` for the UI.
- The project includes `bn_utils.py` with helpers for constructing/updating the `Guilty` CPD; tests live under `tests/`.

**Contributing**
Send PRs against `main`. The CI must pass before merging.
# Bayesian-Network-Dashboard

https://bayesian-network-dashboard-abe.streamlit.app/