"""
Catalog module — loads the normalized catalog + precomputed embeddings at startup
and exposes a search() function combining semantic similarity with keyword boosting.
"""

import json
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Paths relative to project root
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_CATALOG_PATH = os.path.join(_DATA_DIR, "catalog.json")
_EMBEDDINGS_PATH = os.path.join(_DATA_DIR, "embeddings.npy")


@dataclass
class CatalogItem:
    """A single assessment product from the SHL catalog."""
    name: str
    url: str
    test_type: str
    description: str
    job_levels: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    duration_minutes: Optional[int] = None
    remote_testing: bool = True
    adaptive_irt: bool = False


class CatalogStore:
    """
    In-memory catalog store with semantic + keyword search.

    Loaded once at process startup via load(). All access is read-only
    after initialization, so it's thread-safe without locks.
    """

    def __init__(self):
        self.items: List[CatalogItem] = []
        self.embeddings: Optional[np.ndarray] = None
        self._name_to_item: Dict[str, CatalogItem] = {}
        self._url_to_item: Dict[str, CatalogItem] = {}
        self._embed_model = None
        self._loaded = False

    def load(self):
        """Load catalog.json and embeddings.npy from the data directory."""
        logger.info("Loading catalog from %s", _CATALOG_PATH)

        with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        self.items = []
        for record in raw_items:
            item = CatalogItem(
                name=record["name"],
                url=record["url"],
                test_type=record.get("test_type", ""),
                description=record.get("description", ""),
                job_levels=record.get("job_levels", []),
                languages=record.get("languages", []),
                duration_minutes=record.get("duration_minutes"),
                remote_testing=record.get("remote_testing", True),
                adaptive_irt=record.get("adaptive_irt", False),
            )
            self.items.append(item)
            self._name_to_item[item.name] = item
            self._url_to_item[item.url] = item

        logger.info("Loaded %d catalog items", len(self.items))

        # Load precomputed embeddings
        logger.info("Loading embeddings from %s", _EMBEDDINGS_PATH)
        self.embeddings = np.load(_EMBEDDINGS_PATH)
        logger.info("Embeddings shape: %s", self.embeddings.shape)

        # Load the embedding model for query-time embedding
        logger.info("Loading embedding model...")
        from sentence_transformers import SentenceTransformer, CrossEncoder
        self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded")
        
        # Load the cross-encoder for reranking
        self._enable_reranker = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
        if self._enable_reranker:
            logger.info("Loading cross-encoder model (BAAI/bge-reranker-base)...")
            # Using bge-reranker-base as requested for better semantic retrieval
            self._cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
            logger.info("Cross-encoder model loaded")

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def validate_recommendation(self, name: str, url: str) -> Optional[CatalogItem]:
        """
        Check if a (name, url) pair exists verbatim in the catalog.
        Returns the matching CatalogItem if found, None otherwise.

        This is the critical anti-hallucination check — every recommendation
        must pass through this before being included in a response.
        """
        # Try exact name match first
        item = self._name_to_item.get(name)
        if item and item.url == url:
            return item

        # Try exact URL match (in case name has minor differences)
        item = self._url_to_item.get(url)
        if item:
            return item

        # Try URL match with minor normalization (trailing slash)
        normalized_url = url.rstrip("/") + "/"
        item = self._url_to_item.get(normalized_url)
        if item:
            return item

        # Try name-only match as last resort (URL might have a typo from LLM)
        item = self._name_to_item.get(name)
        if item:
            return item

        return None

    def search(self, query: str, top_k: int = 40, mode: str = "rrf") -> List[CatalogItem]:
        """Search the catalog using hybrid retrieval."""
        if not self._loaded or self.embeddings is None or self._embed_model is None:
            logger.warning("Catalog not loaded, returning empty results")
            return []

        n = len(self.items)
        if n == 0:
            return []

        # ── Dense ranking (semantic similarity) ────────────────────────
        query_embedding = self._embed_model.encode(query, normalize_embeddings=True)
        import numpy as np
        dense_scores = np.dot(self.embeddings, query_embedding)

        dense_ranking = np.argsort(dense_scores)[::-1]
        dense_rank = np.zeros(n, dtype=int)
        for rank, idx in enumerate(dense_ranking):
            dense_rank[idx] = rank

        # ── Sparse ranking (keyword/token overlap) ─────────────────────
        query_lower = query.lower()
        import re as _re
        query_tokens = set(_re.findall(r'[a-z0-9]+', query_lower))
        query_tokens = {t for t in query_tokens if len(t) >= 2}

        sparse_scores = np.zeros(n)
        for i, item in enumerate(self.items):
            name_lower = item.name.lower()
            desc_lower = item.description.lower()
            name_tokens = set(_re.findall(r'[a-z0-9]+', name_lower))

            score = 0.0
            for qt in query_tokens:
                if len(qt) >= 3 and qt in name_lower:
                    score += 3.0
            overlap = query_tokens & name_tokens
            score += 2.0 * len(overlap)
            for qt in query_tokens:
                if len(qt) >= 3 and qt in desc_lower:
                    score += 0.5
            sparse_scores[i] = score

        sparse_ranking = np.argsort(sparse_scores)[::-1]
        sparse_rank = np.zeros(n, dtype=int)
        for rank, idx in enumerate(sparse_ranking):
            sparse_rank[idx] = rank

        # ── Mode Selection ─────────────────────────────────────────────
        if mode == "dense":
            top_indices = dense_ranking[:top_k]
            return [self.items[i] for i in top_indices]
        
        elif mode == "sparse":
            top_indices = sparse_ranking[:top_k]
            return [self.items[i] for i in top_indices]
            
        elif mode == "rrf":
            k = 60
            rrf_scores = 1.0 / (k + dense_rank) + 1.0 / (k + sparse_rank)
            top_indices = np.argsort(rrf_scores)[::-1][:top_k]
            return [self.items[i] for i in top_indices]
            
        elif mode == "rrf_stratified":
            k = 60
            rrf_scores = 1.0 / (k + dense_rank) + 1.0 / (k + sparse_rank)
            sorted_indices = np.argsort(rrf_scores)[::-1]
            
            candidates = []
            candidate_urls = set()
            
            # Main ranking
            for i in sorted_indices[:top_k]:
                item = self.items[i]
                candidates.append(item)
                candidate_urls.add(item.url)
                
            # Stratification: Top 2 per category
            categories_found = {cat: 0 for cat in ['A', 'B', 'C', 'D', 'E', 'K', 'P', 'S']}
            for item in candidates:
                for t in item.test_type.split(","):
                    t = t.strip()
                    if t in categories_found:
                        categories_found[t] += 1
                        
            for i in sorted_indices:
                item = self.items[i]
                if item.url in candidate_urls:
                    continue
                item_types = [t.strip() for t in item.test_type.split(",") if t.strip()]
                added = False
                for t in item_types:
                    if t in categories_found and categories_found[t] < 2:
                        if not added:
                            candidates.append(item)
                            candidate_urls.add(item.url)
                            added = True
                        categories_found[t] += 1
                        
            return candidates
            
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def rerank(self, query: str, candidates: List[CatalogItem], top_k: int = 40) -> Tuple[List[CatalogItem], List[str]]:
        """
        Rerank candidates using a cross-encoder.
        Returns (reranked_candidates, debug_candidate_urls)
        """
        if not getattr(self, "_enable_reranker", False) or not hasattr(self, "_cross_encoder"):
            return candidates[:top_k], [c.url for c in candidates]
            
        if len(candidates) <= 40:
            logger.info("Skipping reranking, candidate pool size %d <= 40", len(candidates))
            return candidates[:top_k], [c.url for c in candidates]
            
        import time
        t0 = time.time()
        
        pairs = [[query, f"{c.name} {c.description}"] for c in candidates]
        scores = self._cross_encoder.predict(pairs)
        
        import numpy as np
        sorted_indices = np.argsort(scores)[::-1]
        
        # Log internal scores for diagnostics
        logger.info(f"--- Reranker Diagnostics (Query: '{query[:50]}...') ---")
        for rank, idx in enumerate(sorted_indices[:10], 1):
            c = candidates[idx]
            # We don't have the original dense/sparse here easily without recomputing,
            # but we can log the cross-encoder score and final rank.
            logger.info(f"Rank {rank} | CE Score: {scores[idx]:.4f} | {c.name}")
        logger.info("-----------------------------------------------------")
            
        reranked_candidates = [candidates[i] for i in sorted_indices]
        candidate_urls = [c.url for c in reranked_candidates]
        
        t1 = time.time()
        logger.info("Reranked %d candidates in %.2fs", len(candidates), t1 - t0)
        
        return reranked_candidates[:top_k], candidate_urls

    def get_item_context(self, item: CatalogItem) -> str:
        """Format a catalog item as context for the LLM prompt."""
        parts = [
            f"Name: {item.name}",
            f"URL: {item.url}",
            f"Test Type: {item.test_type}",
            f"Description: {item.description}",
        ]
        if item.job_levels:
            parts.append(f"Job Levels: {', '.join(item.job_levels)}")
        if item.languages:
            lang_str = ", ".join(item.languages[:5])
            if len(item.languages) > 5:
                lang_str += f" (+{len(item.languages) - 5} more)"
            parts.append(f"Languages: {lang_str}")
        if item.duration_minutes is not None:
            parts.append(f"Duration: {item.duration_minutes} minutes")
        else:
            parts.append("Duration: —")
        parts.append(f"Remote Testing: {'Yes' if item.remote_testing else 'No'}")
        parts.append(f"Adaptive/IRT: {'Yes' if item.adaptive_irt else 'No'}")
        return "\n".join(parts)


# Module-level singleton
catalog_store = CatalogStore()
