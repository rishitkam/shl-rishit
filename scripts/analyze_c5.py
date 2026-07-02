import json
import numpy as np
from app.catalog import CatalogStore
from app.agent import _build_retrieval_query

store = CatalogStore()
store.load()

query = "As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?"
query_emb = store._embed_model.encode(query)

# dense scores
dense_scores = np.dot(store.embeddings, query_emb)
dense_ranks = np.argsort(dense_scores)[::-1]

# sparse scores
query_tokens = set(query.lower().split())
sparse_scores = np.zeros(len(store.items))
for i, item in enumerate(store.items):
    item_tokens = set((item.name + " " + item.description).lower().split())
    sparse_scores[i] = len(query_tokens & item_tokens)
sparse_ranks = np.argsort(sparse_scores)[::-1]

# RRF scores
k = 60
rrf_scores = np.zeros(len(store.items))
for rank, idx in enumerate(dense_ranks):
    rrf_scores[idx] += 1.0 / (k + rank + 1)
for rank, idx in enumerate(sparse_ranks):
    rrf_scores[idx] += 1.0 / (k + rank + 1)
rrf_ranks = np.argsort(rrf_scores)[::-1]

targets = ["Global Skills Assessment", "Occupational Personality Questionnaire OPQ32r"]

print(f"Query: {query}\nTokens: {query_tokens}\n")

for target in targets:
    idx = next(i for i, item in enumerate(store.items) if item.name == target)
    dense_r = np.where(dense_ranks == idx)[0][0] + 1
    sparse_r = np.where(sparse_ranks == idx)[0][0] + 1
    rrf_r = np.where(rrf_ranks == idx)[0][0] + 1
    
    print(f"Target: {target}")
    print(f"  Dense Rank:  {dense_r} (Score: {dense_scores[idx]:.4f})")
    print(f"  Sparse Rank: {sparse_r} (Score: {sparse_scores[idx]})")
    print(f"  RRF Rank:    {rrf_r} (Score: {rrf_scores[idx]:.4f})")
    item_tokens = set((store.items[idx].name + " " + store.items[idx].description).lower().split())
    print(f"  Overlapping Tokens: {query_tokens & item_tokens}")
    print()

print("Top 5 Dense:")
for i in range(5):
    idx = dense_ranks[i]
    print(f"  {i+1}. {store.items[idx].name} ({dense_scores[idx]:.4f})")

print("\nTop 5 Sparse:")
for i in range(5):
    idx = sparse_ranks[i]
    print(f"  {i+1}. {store.items[idx].name} ({sparse_scores[idx]})")
