"""Semantic embedding engine for repository files, functions, and classes.

Uses ``fastembed`` (lightweight, no PyTorch) by default with
``all-MiniLM-L6-v2``.  Falls back to a simple TF-IDF-like approach
if fastembed is not installed.  Embeddings are cached in the
``MemoryStore`` and incrementally updated when files change.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from issue_resolver.intelligence.knowledge_graph import RepoKnowledgeGraph


# ---------------------------------------------------------------------------
# Embedding backend abstraction
# ---------------------------------------------------------------------------

_EMBED_MODEL: Any = None
_EMBED_DIM: int = 384  # MiniLM default


def _get_embedding_model() -> Any:
    """Lazily load the embedding model."""
    global _EMBED_MODEL, _EMBED_DIM
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL

    try:
        from fastembed import TextEmbedding
        _EMBED_MODEL = TextEmbedding("BAAI/bge-small-en-v1.5")
        _EMBED_DIM = 384
        print("[Embeddings] Using fastembed (BAAI/bge-small-en-v1.5)")
        return _EMBED_MODEL
    except ImportError:
        pass

    # Fallback: simple bag-of-words hashing (no external deps)
    _EMBED_MODEL = "bow_fallback"
    _EMBED_DIM = 256
    print("[Embeddings] fastembed not available — using BoW fallback")
    return _EMBED_MODEL


def _bow_embed(text: str, dim: int = 256) -> np.ndarray:
    """Simple bag-of-words hash embedding (deterministic, no ML)."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string into a vector."""
    model = _get_embedding_model()
    if model == "bow_fallback":
        return _bow_embed(text, _EMBED_DIM)
    # fastembed returns a generator
    embeddings = list(model.embed([text]))
    return np.array(embeddings[0], dtype=np.float32)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts into a matrix (N x dim)."""
    if not texts:
        return np.zeros((0, _EMBED_DIM), dtype=np.float32)

    model = _get_embedding_model()
    if model == "bow_fallback":
        return np.array([_bow_embed(t, _EMBED_DIM) for t in texts], dtype=np.float32)

    embeddings = list(model.embed(texts))
    return np.array(embeddings, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarities between a query vector and a matrix of vectors."""
    if matrix.shape[0] == 0:
        return np.array([], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    return np.dot(matrix, query) / (norms * query_norm)


# ---------------------------------------------------------------------------
# Repository embedding index
# ---------------------------------------------------------------------------


class RepoEmbeddingIndex:
    """Embeds and indexes all files, functions, and classes in a knowledge graph.

    Usage::

        index = RepoEmbeddingIndex()
        index.build_from_graph(graph)
        results = index.search("calculate total price", top_k=5)
    """

    def __init__(self) -> None:
        self.file_texts: list[str] = []
        self.file_paths: list[str] = []
        self.file_embeddings: np.ndarray | None = None

        self.symbol_texts: list[str] = []
        self.symbol_keys: list[str] = []      # qualified names
        self.symbol_paths: list[str] = []     # file paths
        self.symbol_embeddings: np.ndarray | None = None

    def build_from_graph(self, graph: RepoKnowledgeGraph) -> None:
        """Build embedding index from a knowledge graph."""
        print("[EmbeddingIndex] Building semantic index...")

        # File-level embeddings
        self.file_texts = []
        self.file_paths = []
        for path, module in sorted(graph.modules.items()):
            if module.is_config:
                continue
            text_parts = [f"File: {path}", f"Language: {module.language}"]
            if module.docstring:
                text_parts.append(module.docstring)
            if module.classes:
                text_parts.append(f"Classes: {', '.join(module.classes)}")
            if module.functions:
                text_parts.append(f"Functions: {', '.join(module.functions[:20])}")
            self.file_texts.append("\n".join(text_parts))
            self.file_paths.append(path)

        if self.file_texts:
            self.file_embeddings = embed_batch(self.file_texts)
            print(f"[EmbeddingIndex] Embedded {len(self.file_texts)} files")

        # Symbol-level embeddings (classes + functions)
        self.symbol_texts = []
        self.symbol_keys = []
        self.symbol_paths = []

        for qn, cls in graph.classes.items():
            text = f"class {cls.name}"
            if cls.bases:
                text += f" extends {', '.join(cls.bases)}"
            if cls.docstring:
                text += f": {cls.docstring}"
            if cls.methods:
                text += f" methods: {', '.join(cls.methods[:10])}"
            self.symbol_texts.append(text)
            self.symbol_keys.append(qn)
            self.symbol_paths.append(cls.file_path)

        for qn, fn in graph.functions.items():
            text = f"function {fn.name}"
            if fn.parameters:
                text += f"({', '.join(fn.parameters[:8])})"
            if fn.return_type:
                text += f" -> {fn.return_type}"
            if fn.docstring:
                text += f": {fn.docstring}"
            self.symbol_texts.append(text)
            self.symbol_keys.append(qn)
            self.symbol_paths.append(fn.file_path)

        if self.symbol_texts:
            self.symbol_embeddings = embed_batch(self.symbol_texts)
            print(f"[EmbeddingIndex] Embedded {len(self.symbol_texts)} symbols")

    def search_files(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search files by semantic similarity.  Returns ``[(path, score)]``."""
        if self.file_embeddings is None or len(self.file_paths) == 0:
            return []
        query_vec = embed_text(query)
        scores = cosine_similarity_batch(query_vec, self.file_embeddings)
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self.file_paths[i], float(scores[i])) for i in indices if scores[i] > 0]

    def search_symbols(self, query: str, top_k: int = 10) -> list[tuple[str, str, float]]:
        """Search symbols.  Returns ``[(qualified_name, file_path, score)]``."""
        if self.symbol_embeddings is None or len(self.symbol_keys) == 0:
            return []
        query_vec = embed_text(query)
        scores = cosine_similarity_batch(query_vec, self.symbol_embeddings)
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self.symbol_keys[i], self.symbol_paths[i], float(scores[i]))
            for i in indices if scores[i] > 0
        ]
