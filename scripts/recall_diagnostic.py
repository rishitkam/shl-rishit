"""
Recall Funnel Diagnostic — Step 1 & Step 2

For every expected item in every trace, determines which stage it fails at:
  Stage A: Does it exist in catalog.json at all?
  Stage B: Does it appear in the retrieval candidate set sent to the LLM?
  Stage C: Does the LLM include it in recommendations?

Also audits the matching logic (Step 2): checks for false negatives from
exact-name matching vs URL-based matching.
"""

import json
import os
import re
import sys
from typing import List, Dict, Optional, Tuple
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRACES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "traces", "GenAI_SampleConversations")
CATALOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "catalog.json")


def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    return records


def parse_trace(filepath: str) -> List[Dict]:
    """Parse trace, extracting user and assistant turns with recommendations."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    turns = []
    turn_blocks = re.split(r"### Turn \d+", text)

    for block in turn_blocks[1:]:
        lines = block.strip().split("\n")
        user_content = []
        assistant_content = []
        recommendations = []
        end_of_conv = False
        current_role = None
        in_table = False

        for line in lines:
            if line.strip() == "**User**":
                current_role = "user"
                continue
            elif line.strip() == "**Agent**":
                current_role = "assistant"
                continue

            if current_role == "user":
                if line.strip().startswith(">"):
                    user_content.append(line.strip().lstrip("> "))
            elif current_role == "assistant":
                if "`end_of_conversation`:" in line:
                    end_of_conv = "**true**" in line.lower()
                    continue
                if "No recommendations this turn" in line:
                    continue
                if line.strip().startswith("| #") or line.strip().startswith("|---"):
                    in_table = True
                    continue
                if in_table and line.strip().startswith("|"):
                    cols = [c.strip() for c in line.split("|")[1:-1]]
                    if len(cols) >= 7:
                        name = cols[1].strip()
                        test_type = cols[2].strip()
                        url_match = re.search(r"<(https?://[^>]+)>", cols[6])
                        url = url_match.group(1) if url_match else cols[6].strip()
                        if name and url:
                            recommendations.append({
                                "name": name,
                                "url": url,
                                "test_type": test_type,
                            })
                    continue
                else:
                    in_table = False

        if user_content:
            turns.append({
                "role": "user",
                "content": "\n".join(user_content),
                "recommendations": [],
            })
        if assistant_content or recommendations:
            turns.append({
                "role": "assistant",
                "content": "\n".join(assistant_content),
                "recommendations": recommendations,
            })

    return turns


def url_slug(url: str) -> str:
    """Extract the slug from an SHL catalog URL for matching."""
    url = url.rstrip("/")
    return url.split("/")[-1].lower()


def normalize_name(name: str) -> str:
    """Normalize a name for fuzzy matching."""
    name = name.lower()
    name = re.sub(r'\([^)]*\)', '', name)  # remove parentheticals
    name = re.sub(r'[^a-z0-9\s]', '', name)  # remove punctuation
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def main():
    catalog = load_catalog()
    
    # Build lookup structures
    catalog_by_url = {}
    catalog_by_slug = {}
    catalog_by_name = {}
    catalog_by_normalized_name = {}
    
    for item in catalog:
        url = item["url"].rstrip("/")
        catalog_by_url[url] = item
        catalog_by_url[url + "/"] = item
        slug = url_slug(item["url"])
        catalog_by_slug[slug] = item
        catalog_by_name[item["name"]] = item
        norm = normalize_name(item["name"])
        catalog_by_normalized_name[norm] = item

    print(f"Catalog: {len(catalog)} items")
    print(f"Unique slugs: {len(catalog_by_slug)}")
    print()

    # ──────────────────────────────────────────────────────────────────
    # Step 2: Audit matching logic
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 2: MATCHING AUDIT")
    print("=" * 70)
    
    trace_files = sorted(
        [f for f in os.listdir(TRACES_DIR) if f.endswith(".md")],
        key=lambda x: int(re.search(r'\d+', x).group()),
    )
    
    all_expected = []  # (trace_name, name, url)
    
    for trace_file in trace_files:
        filepath = os.path.join(TRACES_DIR, trace_file)
        trace_name = trace_file.replace(".md", "")
        turns = parse_trace(filepath)
        
        # Get final expected recommendations
        expected_turns = [t for t in turns if t["role"] == "assistant"]
        final_expected = []
        for t in reversed(expected_turns):
            if t["recommendations"]:
                final_expected = t["recommendations"]
                break
        
        for rec in final_expected:
            all_expected.append((trace_name, rec["name"], rec["url"]))
    
    print(f"\nTotal expected items across all traces: {len(all_expected)}\n")
    
    # Check each expected item
    matching_issues = []
    for trace_name, exp_name, exp_url in all_expected:
        exp_slug = url_slug(exp_url)
        
        # Method 1: Exact name match
        name_match = exp_name in catalog_by_name
        
        # Method 2: Exact URL match 
        url_match = (exp_url.rstrip("/") in catalog_by_url or 
                     exp_url.rstrip("/") + "/" in catalog_by_url)
        
        # Method 3: Slug match
        slug_match = exp_slug in catalog_by_slug
        
        # Method 4: Normalized name match
        norm_match = normalize_name(exp_name) in catalog_by_normalized_name
        
        if not name_match and (url_match or slug_match or norm_match):
            matching_issues.append({
                "trace": trace_name,
                "expected_name": exp_name,
                "expected_url": exp_url,
                "name_match": name_match,
                "url_match": url_match,
                "slug_match": slug_match,
                "norm_match": norm_match,
            })
            # Find what catalog has for this
            cat_item = catalog_by_slug.get(exp_slug) or catalog_by_url.get(exp_url.rstrip("/"))
            if cat_item:
                print(f"  ⚠ FALSE NEGATIVE: {trace_name}")
                print(f"    Expected name: '{exp_name}'")
                print(f"    Catalog name:  '{cat_item['name']}'")
                print(f"    URL match: {url_match}, Slug match: {slug_match}")
                print()
        
        if not name_match and not url_match and not slug_match and not norm_match:
            print(f"  ✗ MISSING FROM CATALOG: {trace_name}")
            print(f"    Name: '{exp_name}'")
            print(f"    URL:  '{exp_url}'")
            print(f"    Slug: '{exp_slug}'")
            # Try fuzzy search
            norm_exp = normalize_name(exp_name)
            close = []
            for cat_name, cat_item in catalog_by_name.items():
                if norm_exp in normalize_name(cat_name) or normalize_name(cat_name) in norm_exp:
                    close.append(cat_name)
            if close:
                print(f"    Possible matches: {close}")
            print()
    
    if matching_issues:
        print(f"\n>>> {len(matching_issues)} FALSE NEGATIVES found in name matching!")
        print(">>> These items exist in catalog but would fail exact-name recall check.\n")
    else:
        print("\n>>> No false negatives in name matching.\n")
    
    # ──────────────────────────────────────────────────────────────────
    # Step 1: Stage A check — does each expected item exist in catalog?
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1: RECALL FUNNEL (Stage A — Catalog existence)")
    print("=" * 70)
    
    stage_a_results = {}  # (trace, name) -> catalog_item or None
    
    for trace_name, exp_name, exp_url in all_expected:
        exp_slug = url_slug(exp_url)
        
        # Use URL/slug as primary match (per Step 2 recommendation)
        cat_item = (catalog_by_url.get(exp_url.rstrip("/")) or
                    catalog_by_url.get(exp_url.rstrip("/") + "/") or
                    catalog_by_slug.get(exp_slug))
        
        stage_a_results[(trace_name, exp_name)] = cat_item
    
    a_pass = sum(1 for v in stage_a_results.values() if v is not None)
    a_fail = sum(1 for v in stage_a_results.values() if v is None)
    print(f"\nStage A: {a_pass}/{len(stage_a_results)} expected items exist in catalog")
    print(f"Stage A failures: {a_fail}")
    
    if a_fail > 0:
        print("\nItems missing from catalog:")
        for (trace, name), item in stage_a_results.items():
            if item is None:
                exp_url = [u for t, n, u in all_expected if t == trace and n == name][0]
                print(f"  {trace}: '{name}' ({exp_url})")
    
    print()
    
    # ──────────────────────────────────────────────────────────────────
    # Step 1: Stage B check — does each expected item appear in retrieval?
    # We need to actually run retrieval to test this.
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1: RECALL FUNNEL (Stage B — Retrieval)")
    print("=" * 70)
    print("Loading catalog store for retrieval testing...")
    
    from dotenv import load_dotenv
    load_dotenv()
    from app.catalog import catalog_store
    catalog_store.load()
    
    stage_b_results = {}  # (trace, name) -> bool (found in candidates)
    retrieval_details = {}  # trace -> candidate set info
    
    for trace_file in trace_files:
        filepath = os.path.join(TRACES_DIR, trace_file)
        trace_name = trace_file.replace(".md", "")
        turns = parse_trace(filepath)
        
        expected_turns = [t for t in turns if t["role"] == "assistant"]
        final_expected = []
        for t in reversed(expected_turns):
            if t["recommendations"]:
                final_expected = t["recommendations"]
                break
        
        if not final_expected:
            continue
        
        # Build retrieval query from all user messages (same as agent.py)
        user_texts = [t["content"] for t in turns if t["role"] == "user"]
        query = " ".join(user_texts)
        
        # Run retrieval (same top_k as agent.py)
        candidates = catalog_store.search(query, top_k=30)
        candidate_urls = {c.url.rstrip("/") for c in candidates}
        candidate_slugs = {url_slug(c.url) for c in candidates}
        candidate_names = {c.name for c in candidates}
        
        retrieval_details[trace_name] = {
            "num_candidates": len(candidates),
            "candidate_names": [c.name for c in candidates],
        }
        
        for rec in final_expected:
            exp_slug = url_slug(rec["url"])
            found = (rec["url"].rstrip("/") in candidate_urls or
                     exp_slug in candidate_slugs or
                     rec["name"] in candidate_names)
            stage_b_results[(trace_name, rec["name"])] = found
    
    b_pass = sum(1 for v in stage_b_results.values() if v)
    b_fail = sum(1 for v in stage_b_results.values() if not v)
    print(f"\nStage B: {b_pass}/{len(stage_b_results)} expected items found in retrieval candidates")
    print(f"Stage B failures (cleared A, failed B): {b_fail}")
    
    if b_fail > 0:
        print("\nItems NOT retrieved (but exist in catalog):")
        for (trace, name), found in stage_b_results.items():
            if not found:
                # Check if it's a Stage A failure too
                if stage_a_results.get((trace, name)) is not None:
                    print(f"  {trace}: '{name}'")
    
    print()
    
    # ──────────────────────────────────────────────────────────────────
    # Full funnel table
    # ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("FULL FUNNEL TABLE")
    print("=" * 70)
    
    stage_counts = Counter()  # "A_fail", "B_fail", "C_unknown"
    
    for trace_file in trace_files:
        trace_name = trace_file.replace(".md", "")
        
        # Get expected items for this trace
        trace_expected = [(n, u) for t, n, u in all_expected if t == trace_name]
        
        if not trace_expected:
            continue
        
        print(f"\n--- {trace_name} ---")
        print(f"  Retrieval candidate set size: {retrieval_details.get(trace_name, {}).get('num_candidates', '?')}")
        
        for exp_name, exp_url in trace_expected:
            a_ok = stage_a_results.get((trace_name, exp_name)) is not None
            b_ok = stage_b_results.get((trace_name, exp_name), False)
            
            if not a_ok:
                stage = "A_FAIL (not in catalog)"
                stage_counts["A_fail"] += 1
            elif not b_ok:
                stage = "B_FAIL (not retrieved)"
                stage_counts["B_fail"] += 1
            else:
                stage = "B_PASS (retrieved → goes to LLM)"
                stage_counts["B_pass"] += 1
            
            print(f"  {exp_name:55s} → {stage}")
    
    # ──────────────────────────────────────────────────────────────────
    # Aggregate
    # ──────────────────────────────────────────────────────────────────
    total = len(all_expected)
    print(f"\n{'=' * 70}")
    print(f"AGGREGATE FUNNEL")
    print(f"{'=' * 70}")
    print(f"  Total expected items: {total}")
    print(f"  Stage A failures (not in catalog):  {stage_counts['A_fail']} ({100*stage_counts['A_fail']/total:.0f}%)")
    print(f"  Stage B failures (not retrieved):   {stage_counts['B_fail']} ({100*stage_counts['B_fail']/total:.0f}%)")
    print(f"  Stage B passes (retrieved → LLM):   {stage_counts['B_pass']} ({100*stage_counts['B_pass']/total:.0f}%)")
    print(f"  (Stage C = depends on LLM behavior, tested by live eval harness)")
    print()
    
    # Also check: what retrieval top_k is actually reaching the LLM
    # (accounting for the history injection in agent.py)
    print("=" * 70)
    print("RETRIEVAL DETAILS PER TRACE")
    print("=" * 70)
    for trace_name, details in retrieval_details.items():
        print(f"\n  {trace_name}: {details['num_candidates']} candidates retrieved")
        # Print first few
        for i, name in enumerate(details['candidate_names'][:5]):
            print(f"    [{i+1}] {name}")
        if len(details['candidate_names']) > 5:
            print(f"    ... and {len(details['candidate_names']) - 5} more")


if __name__ == "__main__":
    main()
