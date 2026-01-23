import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"
EXPORT_DIR = STORAGE_DIR / "exports"
RUNS_FILE = STORAGE_DIR / "runs.json"

STORAGE_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------
# Load history
# ---------------------------------------------------

def load_runs():
    if RUNS_FILE.exists():
        return json.loads(RUNS_FILE.read_text())
    return []

# ---------------------------------------------------
# Save run
# ---------------------------------------------------

def save_run(payload):
    runs = load_runs()
    payload["timestamp"] = datetime.utcnow().isoformat()
    runs.append(payload)
    RUNS_FILE.write_text(json.dumps(runs, indent=2))
    return payload

# ---------------------------------------------------
# Clear history
# ---------------------------------------------------

def clear_runs():
    RUNS_FILE.write_text("[]")
