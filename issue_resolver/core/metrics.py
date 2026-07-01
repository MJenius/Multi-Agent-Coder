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


def compute_localization_quality_metrics(state: dict, is_resolved: bool) -> dict:
    """Calculate and return updated metrics containing precision, recall, graph hit rate, and calibration error."""
    localization = state.get("localization_result", {})
    metrics = state.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
        
    primary_files = [f["path"] for f in localization.get("primary_files", [])] if isinstance(localization, dict) else []
    
    # Final edited files from structured_plan
    structured_plan = state.get("structured_plan", {})
    final_edited_files = []
    if isinstance(structured_plan, dict):
        final_edited_files = structured_plan.get("files_to_edit", [])
        
    if not final_edited_files and state.get("proposed_fix"):
        import re
        final_edited_files = re.findall(r"^\+\+\+\s+(\S+)", state.get("proposed_fix", ""), re.MULTILINE)
        # Strip potential unified diff prefixes like a/ and b/
        clean_files = []
        for f in final_edited_files:
            # remove a/ or b/ if at the start
            if f.startswith(("a/", "b/")):
                clean_files.append(f[2:])
            else:
                clean_files.append(f)
        final_edited_files = clean_files
        
    # Standardize files to relative path suffix matching
    expected_f = {f.replace("\\", "/").lstrip("./") for f in final_edited_files if f}
    retrieved_f = {f.replace("\\", "/").lstrip("./") for f in primary_files if f}
    
    if expected_f:
        # Match expected files with retrieved files
        # A file is matched if it matches exactly or is a suffix of a retrieved file
        matched_f = set()
        for exp in expected_f:
            for ret in retrieved_f:
                if exp == ret or exp.endswith("/" + ret) or ret.endswith("/" + exp):
                    matched_f.add(exp)
                    break
        loc_recall = len(matched_f) / len(expected_f)
        loc_precision = len(matched_f) / len(retrieved_f) if retrieved_f else 0.0
        all_edited_in_initial = (len(matched_f) == len(expected_f))
    else:
        loc_recall = 0.0
        loc_precision = 0.0
        all_edited_in_initial = False
        
    # Graph hit rate
    graph_hits = localization.get("graph_hits", 0) if isinstance(localization, dict) else 0
    graph_misses = localization.get("graph_misses", 0) if isinstance(localization, dict) else 0
    total_graph_ops = graph_hits + graph_misses
    graph_hit_rate = graph_hits / total_graph_ops if total_graph_ops > 0 else 0.0
    
    # Confidence calibration error
    pred_conf = state.get("localization_confidence", 0.0)
    actual_success = 1.0 if is_resolved else 0.0
    calibration_error = abs(pred_conf - actual_success)
    
    quality_metrics = {
        "precision": loc_precision,
        "recall": loc_recall,
        "all_edited_in_initial": all_edited_in_initial,
        "graph_hit_rate": graph_hit_rate,
        "confidence_calibration_error": calibration_error,
        "predicted_confidence": pred_conf,
        "actual_success": is_resolved,
    }
    
    metrics["localization_quality"] = quality_metrics
    
    # Diagnostic logging (Requirement 6)
    print(f"[Metrics] Diagnostics: Localization quality calculated:")
    print(f"  - Precision: {loc_precision:.2f}, Recall: {loc_recall:.2f}, Graph hit rate: {graph_hit_rate:.2f}")
    print(f"  - Calibration error: {calibration_error:.2f}, All edited files present in initial: {all_edited_in_initial}")
    
    return metrics

