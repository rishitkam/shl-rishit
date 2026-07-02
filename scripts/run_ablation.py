import asyncio
import json
import glob
import os
import sys
import time
import sys
import time
from typing import List, Dict

import dotenv
dotenv.load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.catalog import catalog_store
from app.schemas import ChatRequest, Message
from app.agent import generate_response, _build_retrieval_query
import app.agent as agent_module
from scripts.eval_harness import parse_trace

def load_traces(traces_dir="traces/GenAI_SampleConversations") -> List[Dict]:
    traces = []
    for filepath in sorted(glob.glob(os.path.join(traces_dir, "*.md"))):
        turns = parse_trace(filepath)
        # The expected recommendations are the recommendations from the last assistant turn
        expected = []
        for turn in reversed(turns):
            if turn["role"] == "assistant" and turn["recommendations"]:
                expected = [r["name"] for r in turn["recommendations"]]
                break
        traces.append({"file": os.path.basename(filepath), "turns": turns, "expected": expected})
    return traces

async def run_ablation(mode: str):
    catalog_store.load()
    traces = load_traces()
    
    total_expected = 0
    candidate_hits_5 = 0
    candidate_hits_10 = 0
    candidate_hits_40 = 0
    llm_selection_hits = 0
    grounded_hits = 0
    
    total_latency = 0.0
    num_calls = 0
    
    hallucination_count = 0
    invalid_url_count = 0
    
    # We will mock the validation to capture LLM raw recommendations
    original_validate = agent_module._validate_recommendations
    
    for trace in traces:
        messages = [Message(role=t["role"], content=t["content"]) for t in trace["turns"]]
        
        # 1. Candidate Metrics
        query = _build_retrieval_query(messages)
        candidates = catalog_store.search(query, top_k=40, mode=mode)
        
        exp_set = set(trace["expected"])
        total_expected += len(exp_set)
        
        c_names = [c.name for c in candidates]
        c5_hits = sum(1 for e in exp_set if e in c_names[:5])
        c10_hits = sum(1 for e in exp_set if e in c_names[:10])
        c40_hits = sum(1 for e in exp_set if e in c_names[:40])
        
        candidate_hits_5 += c5_hits
        candidate_hits_10 += c10_hits
        candidate_hits_40 += c40_hits
        
        # 2. LLM + Grounded Metrics
        raw_recs_captured = []
        def _mock_validate(raw_recs):
            raw_recs_captured.extend(raw_recs)
            return original_validate(raw_recs)
            
        agent_module._validate_recommendations = _mock_validate
        
        req = ChatRequest(messages=messages)
        start_t = time.time()
        
        # Temporarily monkeypatch catalog search in the agent to use the mode
        original_search = catalog_store.search
        def _mock_search(q, top_k=40, m=mode):
            return original_search(q, top_k, mode=m)
        catalog_store.search = _mock_search
        
        response = await generate_response(req)
        
        # Restore original search
        catalog_store.search = original_search
        
        total_latency += (time.time() - start_t)
        num_calls += 1
        
        # Restore original
        agent_module._validate_recommendations = original_validate
        
        # Compute LLM Selection Recall (from captured raw recs)
        raw_names = [r.get("name", "") for r in raw_recs_captured if isinstance(r, dict)]
        llm_selection_hits += sum(1 for e in exp_set if e in raw_names)
        
        # Compute Grounded Recall (from final validated response)
        grounded_names = [r.name for r in response.recommendations]
        grounded_hits += sum(1 for e in exp_set if e in grounded_names)
        
        # Check hallucinations in raw recs
        catalog_urls = set(item.url for item in catalog_store.items)
        catalog_names = set(item.name for item in catalog_store.items)
        
        for r in raw_recs_captured:
            if isinstance(r, dict):
                n = r.get("name", "")
                u = r.get("url", "")
                if n not in catalog_names and u not in catalog_urls:
                    hallucination_count += 1
                if u not in catalog_urls:
                    invalid_url_count += 1
                    
        # To avoid 429 quota exhaustion even with backoff, add a static delay between traces
        time.sleep(2.0)

    # Print results
    print(f"\n--- Metrics for {mode.upper()} ---")
    print(f"Recall@10 (Grounded Final): {grounded_hits / max(1, total_expected):.2f}")
    print(f"LLM Selection Recall:       {llm_selection_hits / max(1, total_expected):.2f}")
    print(f"Candidate Recall@40:        {candidate_hits_40 / max(1, total_expected):.2f}")
    print(f"Candidate Recall@10:        {candidate_hits_10 / max(1, total_expected):.2f}")
    print(f"Candidate Recall@5:         {candidate_hits_5 / max(1, total_expected):.2f}")
    print(f"Average Latency:            {total_latency / max(1, num_calls):.2f}s")
    print(f"Hallucination Count:        {hallucination_count}")
    print(f"Invalid URL Count:          {invalid_url_count}")

async def main():
    print("Starting Phase 1 Retrieval Ablation Baseline...")
    for mode in ["dense", "sparse", "rrf", "rrf_stratified"]:
        await run_ablation(mode)
        
if __name__ == "__main__":
    asyncio.run(main())
