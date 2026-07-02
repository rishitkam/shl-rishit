import json
import glob
import os
import asyncio
from typing import List, Dict

# Setup app imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.catalog import catalog_store
from app.schemas import ChatRequest, Message
from app.agent import generate_response, _build_retrieval_query

def load_traces(traces_dir="traces/GenAI_SampleConversations") -> List[Dict]:
    traces = []
    for filepath in sorted(glob.glob(os.path.join(traces_dir, "*.md"))):
        with open(filepath, "r") as f:
            content = f.read()
            
        turns = []
        expected = []
        
        parts = content.split("## Expected Assessments")
        conv_text = parts[0]
        if len(parts) > 1:
            for line in parts[1].strip().split("\n"):
                if line.startswith("- "):
                    expected.append(line.strip("- ").strip())
                    
        for block in conv_text.split("### "):
            if block.startswith("User:"):
                turns.append({"role": "user", "content": block.replace("User:", "").strip()})
            elif block.startswith("Assistant:"):
                turns.append({"role": "assistant", "content": block.replace("Assistant:", "").strip()})
                
        traces.append({"file": os.path.basename(filepath), "turns": turns, "expected": expected})
    return traces

async def run_ablation(mode: str):
    catalog_store.load()
    traces = load_traces()
    
    total_expected = 0
    candidate_hits_5 = 0
    candidate_hits_10 = 0
    candidate_hits_40 = 0
    grounded_hits = 0
    
    print(f"\n--- Running Ablation: {mode} ---")
    
    for trace in traces:
        # Reconstruct exactly what the agent sees at the end of the trace
        messages = [Message(role=t["role"], content=t["content"]) for t in trace["turns"]]
        
        # 1. Candidate Recall
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
        
        # 2. Final Response
        # We can bypass HTTP and call the agent directly to avoid server restarts,
        # but the agent internally calls catalog_store.search which defaults to "rrf".
        # We must monkeypatch or set env var so agent uses the right mode.
        # Actually, the agent hardcodes `catalog_store.search(query, top_k=40)` 
        # so it defaults to "rrf". Let's patch agent.py to use an env var for mode.
        
