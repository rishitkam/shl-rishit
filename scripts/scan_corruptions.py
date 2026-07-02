import json
import re
import html
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "catalog_raw.json"

def is_corrupted(text: str) -> bool:
    if not isinstance(text, str):
        return False
    # Check for newlines, tabs, multiple spaces, html entities, leading/trailing whitespace
    if '\n' in text or '\r' in text or '\t' in text:
        return True
    if '  ' in text:
        return True
    if text != text.strip():
        return True
    if '&amp;' in text or '&lt;' in text or '&gt;' in text or '&quot;' in text or '&#39;' in text:
        return True
    return False

def main():
    raw_text = RAW_PATH.read_text(encoding="utf-8")
    raw_data = json.loads(raw_text, strict=False)
    
    flagged = []
    
    for record in raw_data:
        name = record.get("name", "Unknown")
        issues = []
        for key in ["name", "description"]:
            val = record.get(key, "")
            if is_corrupted(val):
                issues.append(key)
        
        if issues:
            flagged.append((name, issues))
            
    print(f"Total flagged records: {len(flagged)}")
    for name, issues in flagged:
        print(f"Record: {name} | Fields: {', '.join(issues)}")
        
if __name__ == "__main__":
    main()
