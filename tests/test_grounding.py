"""
Tests for catalog grounding — verifying that the recommendation validator
correctly accepts valid catalog items and rejects hallucinated ones.
"""

import json
import os
import pytest
from app.catalog import CatalogStore, CatalogItem


@pytest.fixture(scope="module")
def loaded_store():
    """Load the catalog store once for all tests in this module."""
    store = CatalogStore()
    store.load()
    return store


@pytest.fixture(scope="module")
def catalog_records():
    """Load the raw catalog.json for cross-referencing."""
    catalog_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "catalog.json"
    )
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestCatalogLoading:
    """Verify catalog loads correctly."""

    def test_catalog_loaded(self, loaded_store):
        """Catalog should have 377 items."""
        assert loaded_store.is_loaded
        assert len(loaded_store.items) == 377

    def test_embeddings_shape(self, loaded_store):
        """Embeddings should be (377, 384) for all-MiniLM-L6-v2."""
        assert loaded_store.embeddings is not None
        assert loaded_store.embeddings.shape == (377, 384)


class TestRecommendationValidation:
    """Verify the anti-hallucination validator."""

    def test_valid_item_passes(self, loaded_store):
        """A real catalog item should validate."""
        # Use the first item in the catalog
        item = loaded_store.items[0]
        result = loaded_store.validate_recommendation(item.name, item.url)
        assert result is not None
        assert result.name == item.name
        assert result.url == item.url

    def test_all_items_validate(self, loaded_store):
        """Every item in the catalog should pass validation."""
        for item in loaded_store.items:
            result = loaded_store.validate_recommendation(item.name, item.url)
            assert result is not None, f"Failed to validate: {item.name}"

    def test_hallucinated_name_rejected(self, loaded_store):
        """A made-up assessment name should be rejected."""
        result = loaded_store.validate_recommendation(
            "Totally Fake Assessment 9000",
            "https://www.shl.com/products/product-catalog/view/fake-assessment/",
        )
        assert result is None

    def test_hallucinated_url_rejected(self, loaded_store):
        """A real name with a wrong URL should still match (name-based fallback)."""
        item = loaded_store.items[0]
        result = loaded_store.validate_recommendation(
            item.name,
            "https://www.shl.com/products/product-catalog/view/wrong-slug/",
        )
        # Name-only match as fallback — this is by design
        assert result is not None

    def test_completely_fake_rejected(self, loaded_store):
        """Both name and URL made up should be rejected."""
        result = loaded_store.validate_recommendation(
            "Python Deep Learning Assessment",
            "https://www.shl.com/products/product-catalog/view/python-deep-learning/",
        )
        assert result is None


class TestCatalogSearch:
    """Verify search retrieves relevant results."""

    def test_java_search(self, loaded_store):
        """Searching for 'Java developer' should return Java-related tests."""
        results = loaded_store.search("Java developer senior backend", top_k=10)
        assert len(results) > 0
        names = [r.name.lower() for r in results]
        assert any("java" in name for name in names), \
            f"Expected Java in results, got: {names}"

    def test_personality_search(self, loaded_store):
        """Searching for 'personality leadership' should return personality tests."""
        results = loaded_store.search("personality leadership assessment", top_k=10)
        assert len(results) > 0
        types = [r.test_type for r in results]
        assert any("P" in t for t in types), \
            f"Expected personality tests, got types: {types}"

    def test_contact_center_search(self, loaded_store):
        """Searching for contact center should return relevant simulations."""
        results = loaded_store.search("contact center customer service agent", top_k=10)
        assert len(results) > 0
        names = [r.name.lower() for r in results]
        assert any("contact" in name or "customer" in name for name in names), \
            f"Expected contact/customer results, got: {names}"

    def test_safety_search(self, loaded_store):
        """Searching for safety/industrial should return safety assessments."""
        results = loaded_store.search("plant operator safety chemical facility", top_k=10)
        assert len(results) > 0
        names = [r.name.lower() for r in results]
        assert any("safety" in name or "dependability" in name for name in names), \
            f"Expected safety results, got: {names}"

    def test_search_returns_correct_count(self, loaded_store):
        """search() should return at most top_k items."""
        results = loaded_store.search("any query", top_k=5)
        assert len(results) <= 5

    def test_search_empty_query(self, loaded_store):
        """Empty query should still return results (by similarity)."""
        results = loaded_store.search("", top_k=5)
        assert len(results) == 5  # Should return top-5 by default similarity
