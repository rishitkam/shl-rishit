import asyncio
import json
import numpy as np
import os
import re
from app.catalog import CatalogStore
from app.query_expansion import expand_query_async

from scripts.eval_harness import parse_trace

async def main():
    catalog_store = CatalogStore()
    catalog_store.load()
    
    traces_dir = "traces/GenAI_SampleConversations"
    trace_files = sorted([f for f in os.listdir(traces_dir) if f.endswith(".md")], key=lambda x: int(re.search(r'\d+', x).group()))
    
    for trace_file in trace_files:
        trace_id = trace_file.replace(".md", "")
        filepath = os.path.join(traces_dir, trace_file)
        turns = parse_trace(filepath)
        
        user_turns = [t for t in turns if t["role"] == "user"]
        expected_turns = [t for t in turns if t["role"] == "assistant"]
        
        final_expected = []
        for t in reversed(expected_turns):
            if t["recommendations"]:
                final_expected = t["recommendations"]
                break
                
        # Only look at the final user query for the diagnostic
        last_user_idx = -1
        for i, m in enumerate(turns):
            if m["role"] == "user":
                last_user_idx = i
                
        if last_user_idx == -1:
            continue
            
        history = [{"role": m["role"], "content": m["content"]} for m in turns[:last_user_idx]]
        current_query = turns[last_user_idx]["content"]
        
        # Expand query
        expanded_query = await expand_query_async(current_query)
        
        # Run dense and sparse separately to get scores
        dense_query = expanded_query
        sparse_query = expanded_query
        
        query_emb = catalog_store._embed_model.encode(dense_query, normalize_embeddings=True)
        dense_scores = np.dot(catalog_store.embeddings, query_emb)
        dense_ranking = np.argsort(dense_scores)[::-1]
        
        def _get_sparse_scores(query: str) -> np.ndarray:
            scores = np.zeros(len(catalog_store.items))
            terms = [t.lower() for t in re.findall(r'\w+', query)]
            if not terms:
                return scores
            for i, item in enumerate(catalog_store.items):
                text = (item.name + " " + item.test_type).lower()
                score = sum(1 for t in terms if t in text)
                scores[i] = score
            return scores
            
        sparse_scores = _get_sparse_scores(sparse_query)
        sparse_ranking = np.argsort(sparse_scores)[::-1]
        
        dense_rank_map = {idx: rank for rank, idx in enumerate(dense_ranking)}
        sparse_rank_map = {idx: rank for rank, idx in enumerate(sparse_ranking)}
        
        k = 60
        rrf_scores = np.zeros(len(catalog_store.items))
        for i in range(len(catalog_store.items)):
            rrf_scores[i] = 1.0 / (k + dense_rank_map[i]) + 1.0 / (k + sparse_rank_map[i])
            
        final_ranking = np.argsort(rrf_scores)[::-1]
        
        expected_urls = {r["url"].rstrip("/") for r in final_expected}
        
        print(f"=== Trace {trace_id} ===")
        print(f"Query: {current_query}")
        print(f"Expanded: {expanded_query}")
        
        expected_found = []
        for rank, idx in enumerate(final_ranking):
            item = catalog_store.items[idx]
            if item.url.rstrip("/") in expected_urls:
                expected_found.append({
                    "name": item.name,
                    "rank": rank + 1,
                    "dense_rank": dense_rank_map[idx] + 1,
                    "sparse_rank": sparse_rank_map[idx] + 1,
                    "rrf_score": rrf_scores[idx],
                    "dense_score": dense_scores[idx],
                    "sparse_score": sparse_scores[idx]
                })
                
        print("Expected Assessments:")
        for ef in expected_found:
            print(f"  [{ef['rank']:3d}] {ef['name'][:40]:40s} | DenseRank: {ef['dense_rank']:3d} | SparseRank: {ef['sparse_rank']:3d}")
            
        print("\nTop 5 Irrelevant Assessments (Why are they ranked high?):")
        irrelevant_count = 0
        for rank, idx in enumerate(final_ranking):
            item = catalog_store.items[idx]
            if item.url.rstrip("/") not in expected_urls:
                print(f"  [{rank+1:3d}] {item.name[:40]:40s} | DenseRank: {dense_rank_map[idx]:3d} | SparseRank: {sparse_rank_map[idx]:3d}")
                irrelevant_count += 1
                if irrelevant_count >= 5:
                    break
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())
