"""
build_embeddings.py
Reads data/catalog.json and produces data/embeddings.npy using all-MiniLM-L6-v2.
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = BASE_DIR / "data" / "catalog.json"
EMB_PATH = BASE_DIR / "data" / "embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"


def build_text(record: dict) -> str:
    """Concatenate fields into a single embedding string."""
    parts = [
        record.get("name", ""),
        record.get("description", ""),
        record.get("test_type", ""),
        " ".join(record.get("job_levels", [])),
    ]
    return " ".join(parts)


def main():
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(catalog)} records from {CATALOG_PATH}")

    texts = [build_text(r) for r in catalog]

    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding texts...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype=np.float32)

    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, embeddings)
    print(f"Saved embeddings to {EMB_PATH}")

    # ── Verification ───────────────────────────────────────────────────
    loaded = np.load(EMB_PATH)
    print(f"  Shape: {loaded.shape}")
    print(f"  Dtype: {loaded.dtype}")
    assert loaded.shape == (len(catalog), 384), f"Unexpected shape: {loaded.shape}"
    print("✓ embeddings.npy looks good")


if __name__ == "__main__":
    main()
