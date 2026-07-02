"""
build_catalog.py
Reads data/catalog_raw.json and produces data/catalog.json with normalised schema.
"""
import json
import re
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "catalog_raw.json"
OUT_PATH = BASE_DIR / "data" / "catalog.json"

# ── Key → single-letter test-type code ─────────────────────────────────────
KEY_MAP = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# Duration values that should map to None
NULL_DURATIONS = {"untimed", "variable", "n/a", "tbc", "", "-"}


def parse_duration(raw: str) -> int | None:
    """Return integer minutes or None."""
    raw = raw.strip()
    if raw.lower() in NULL_DURATIONS:
        return None
    m = re.search(r"(\d+)", raw)
    if m:
        val = int(m.group(1))
        return val if val > 0 else None
    return None


import html

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = html.unescape(text)
    # Replace carriage returns, newlines, tabs with spaces
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def normalise(record: dict) -> dict:
    # Build test_type codes from keys list
    codes = sorted(
        {KEY_MAP[k] for k in record.get("keys", []) if k in KEY_MAP}
    )
    test_type = ",".join(codes)

    return {
        "name": clean_text(record.get("name", "")),
        "url": clean_text(record.get("link", "")),
        "test_type": test_type,
        "description": clean_text(record.get("description", "")),
        "job_levels": record.get("job_levels", []),
        "languages": record.get("languages", []),
        "duration_minutes": parse_duration(record.get("duration", "")),
        "remote_testing": record.get("remote", "").lower() == "yes",
        "adaptive_irt": record.get("adaptive", "").lower() == "yes",
    }


def main():
    # strict=False handles invalid control characters in the raw JSON
    raw_text = RAW_PATH.read_text(encoding="utf-8")
    raw_data = json.loads(raw_text, strict=False)
    print(f"Loaded {len(raw_data)} raw records from {RAW_PATH}")

    catalog = [normalise(r) for r in raw_data]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(catalog)} records to {OUT_PATH}")

    # ── Quick verification ─────────────────────────────────────────────
    expected_keys = {
        "name", "url", "test_type", "description",
        "job_levels", "languages", "duration_minutes",
        "remote_testing", "adaptive_irt",
    }
    for i, rec in enumerate(catalog):
        assert set(rec.keys()) == expected_keys, f"Record {i} has wrong keys: {set(rec.keys())}"

    durations = [r["duration_minutes"] for r in catalog if r["duration_minutes"] is not None]
    print(f"  Records with duration: {len(durations)}")
    print(f"  Duration range: {min(durations)}–{max(durations)} minutes")

    types = set()
    for r in catalog:
        types.update(r["test_type"].split(","))
    types.discard("")
    print(f"  Test-type codes found: {sorted(types)}")
    print("✓ catalog.json looks good")


if __name__ == "__main__":
    main()
