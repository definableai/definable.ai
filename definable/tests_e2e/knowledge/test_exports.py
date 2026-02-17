"""Tests for knowledge package exports.

Verifies that embedder and reranker implementations are properly
re-exported from their respective package __init__.py files,
so users can import them without using internal module paths.
"""

import importlib

import pytest


class TestEmbedderExports:
    """Test that embedder implementations are exported from the embedders package."""

    def test_openai_embedder_importable_from_embedders_package(self):
        """OpenAIEmbedder should be importable from definable.knowledge.embedders."""
        mod = importlib.import_module("definable.knowledge.embedders")
        assert hasattr(mod, "OpenAIEmbedder")
        from definable.knowledge.embedders.openai import OpenAIEmbedder

        assert mod.OpenAIEmbedder is OpenAIEmbedder

    def test_voyageai_embedder_importable_from_embedders_package(self):
        """VoyageAIEmbedder should be importable from definable.knowledge.embedders."""
        mod = importlib.import_module("definable.knowledge.embedders")
        assert hasattr(mod, "VoyageAIEmbedder")
        from definable.knowledge.embedders.voyageai import VoyageAIEmbedder

        assert mod.VoyageAIEmbedder is VoyageAIEmbedder

    def test_embedder_base_still_exported(self):
        """Embedder base class should still be directly importable."""
        from definable.knowledge.embedders import Embedder
        from definable.knowledge.embedders.base import Embedder as BaseEmbedder

        assert Embedder is BaseEmbedder

    def test_embedders_all_includes_implementations(self):
        """__all__ should list both base class and implementations."""
        import definable.knowledge.embedders as embedders_pkg

        assert "Embedder" in embedders_pkg.__all__
        assert "OpenAIEmbedder" in embedders_pkg.__all__
        assert "VoyageAIEmbedder" in embedders_pkg.__all__


class TestRerankerExports:
    """Test that reranker implementations are exported from the rerankers package."""

    def test_cohere_reranker_importable_from_rerankers_package(self):
        """CohereReranker should be importable from definable.knowledge.rerankers."""
        pytest.importorskip("cohere")
        mod = importlib.import_module("definable.knowledge.rerankers")
        assert hasattr(mod, "CohereReranker")
        from definable.knowledge.rerankers.cohere import CohereReranker

        assert mod.CohereReranker is CohereReranker

    def test_reranker_base_still_exported(self):
        """Reranker base class should still be directly importable."""
        from definable.knowledge.rerankers import Reranker
        from definable.knowledge.rerankers.base import Reranker as BaseReranker

        assert Reranker is BaseReranker

    def test_rerankers_all_includes_implementations(self):
        """__all__ should list both base class and implementations."""
        import definable.knowledge.rerankers as rerankers_pkg

        assert "Reranker" in rerankers_pkg.__all__
        assert "CohereReranker" in rerankers_pkg.__all__


class TestTopLevelKnowledgeExports:
    """Test that the top-level knowledge package also re-exports implementations."""

    def test_openai_embedder_from_knowledge_package(self):
        """OpenAIEmbedder should be importable from definable.knowledge."""
        mod = importlib.import_module("definable.knowledge")
        assert hasattr(mod, "OpenAIEmbedder")

    def test_voyageai_embedder_from_knowledge_package(self):
        """VoyageAIEmbedder should be importable from definable.knowledge."""
        mod = importlib.import_module("definable.knowledge")
        assert hasattr(mod, "VoyageAIEmbedder")

    def test_cohere_reranker_from_knowledge_package(self):
        """CohereReranker should be importable from definable.knowledge."""
        pytest.importorskip("cohere")
        mod = importlib.import_module("definable.knowledge")
        assert hasattr(mod, "CohereReranker")

    def test_knowledge_all_includes_implementations(self):
        """Top-level __all__ should list embedder and reranker implementations."""
        import definable.knowledge as knowledge_pkg

        assert "OpenAIEmbedder" in knowledge_pkg.__all__
        assert "VoyageAIEmbedder" in knowledge_pkg.__all__
        assert "CohereReranker" in knowledge_pkg.__all__

    def test_invalid_attr_raises_attribute_error(self):
        """Accessing a non-existent attribute should raise AttributeError."""
        mod = importlib.import_module("definable.knowledge.embedders")
        with pytest.raises(AttributeError):
            _ = mod.NonExistentClass

        mod = importlib.import_module("definable.knowledge.rerankers")
        with pytest.raises(AttributeError):
            _ = mod.NonExistentClass
