"""Run-level metrics collection and tracking.

Accumulates and logs telemetry about localization accuracy, execution efficiency,
retries, and token usage for single pipeline runs and benchmark evaluations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunMetrics:
    # Localization metrics
    localization_confidence: float = 0.0
    files_retrieved: int = 0
    symbols_retrieved: int = 0
    graph_hit_rate: float = 0.0
    lsp_available: bool = False
    researcher_fallback_used: bool = False
    classification_method: str = "deterministic"

    # Execution metrics
    verification_success: bool = False
    retry_count: int = 0
    patch_accepted: bool = False
    files_modified: int = 0

    # Efficiency metrics
    total_runtime_ms: float = 0.0
    llm_calls_total: int = 0
    deterministic_ops_total: int = 0
    total_tokens: int = 0
    model_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        return {
            "localization_confidence": self.localization_confidence,
            "files_retrieved": self.files_retrieved,
            "symbols_retrieved": self.symbols_retrieved,
            "graph_hit_rate": self.graph_hit_rate,
            "lsp_available": self.lsp_available,
            "researcher_fallback_used": self.researcher_fallback_used,
            "classification_method": self.classification_method,
            "verification_success": self.verification_success,
            "retry_count": self.retry_count,
            "patch_accepted": self.patch_accepted,
            "files_modified": self.files_modified,
            "total_runtime_ms": self.total_runtime_ms,
            "llm_calls_total": self.llm_calls_total,
            "deterministic_ops_total": self.deterministic_ops_total,
            "total_tokens": self.total_tokens,
            "model_usage": self.model_usage,
        }

    def log_summary(self) -> None:
        """Print a summary of run metrics to console."""
        print("=" * 60)
        print("                       RUN METRICS SUMMARY                      ")
        print("=" * 60)
        print(f"Classification method:     {self.classification_method}")
        print(f"LSP bridge available:      {self.lsp_available}")
        print(f"Researcher fallback used:  {self.researcher_fallback_used}")
        print(f"Localization confidence:   {self.localization_confidence:.2f}")
        print(f"Files / Symbols found:     {self.files_retrieved} / {self.symbols_retrieved}")
        print(f"Graph hit rate:            {self.graph_hit_rate:.2f}")
        print("-" * 60)
        print(f"Verification success:      {self.verification_success}")
        print(f"Files modified:            {self.files_modified}")
        print(f"Retry attempts:            {self.retry_count}")
        print(f"Patch accepted:            {self.patch_accepted}")
        print("-" * 60)
        print(f"Total runtime:             {self.total_runtime_ms/1000:.2f} seconds")
        print(f"LLM Calls:                 {self.llm_calls_total}")
        print(f"Deterministic Operations:  {self.deterministic_ops_total}")
        print(f"Total Tokens:              {self.total_tokens}")
        if self.model_usage:
            print("Model token usage breakdown:")
            for model, tokens in self.model_usage.items():
                print(f"  - {model}: {tokens} tokens")
        print("=" * 60)
