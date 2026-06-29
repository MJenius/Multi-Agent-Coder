"""Structured execution trace for every agent run.

Records agent decisions, selected models, retrieved context, verification
results, retry history, and timing.  Does **not** log chain-of-thought
or raw LLM responses.  The trace is written to a JSON file for each run
and can be used for debugging, benchmarking, and auditing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    """A single event in the execution trace."""

    timestamp: str
    event_type: str          # "agent_decision", "model_selection", "retrieval", etc.
    agent: str               # which agent produced this event
    action: str              # what happened
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    tokens_used: int = 0
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExecutionTrace:
    """Accumulates trace events for a single pipeline run.

    Usage::

        trace = ExecutionTrace(run_id="abc123")
        trace.record("agent_decision", "Planner", "selected_model",
                      details={"model": "glm-5.1", "reason": "default"})
        trace.save("./traces/")
    """

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.start_time = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.events: list[TraceEvent] = []
        self.summary: dict[str, Any] = {}

    # ----- recording -----

    def record(
        self,
        event_type: str,
        agent: str,
        action: str,
        *,
        details: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
        tokens_used: int = 0,
        model: str = "",
    ) -> TraceEvent:
        """Append a new event to the trace."""
        event = TraceEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            agent=agent,
            action=action,
            details=details or {},
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            model=model,
        )
        self.events.append(event)
        return event

    def record_model_selection(
        self,
        agent: str,
        role: str,
        model: str,
        reason: str = "",
    ) -> TraceEvent:
        """Convenience: record a model selection decision."""
        return self.record(
            "model_selection",
            agent,
            f"selected {model} for {role}",
            details={"role": role, "model": model, "reason": reason},
            model=model,
        )

    def record_retrieval(
        self,
        agent: str,
        files: list[str],
        confidence_scores: dict[str, float] | None = None,
    ) -> TraceEvent:
        """Convenience: record which files were retrieved and their scores."""
        return self.record(
            "retrieval",
            agent,
            f"retrieved {len(files)} files",
            details={
                "files": files,
                "confidence_scores": confidence_scores or {},
            },
        )

    def record_verification(
        self,
        step_name: str,
        passed: bool,
        output_snippet: str = "",
        duration_ms: float = 0.0,
    ) -> TraceEvent:
        """Convenience: record a verification step result."""
        return self.record(
            "verification",
            "verifier",
            f"{step_name}: {'PASS' if passed else 'FAIL'}",
            details={"step": step_name, "passed": passed, "output": output_snippet[:500]},
            duration_ms=duration_ms,
        )

    def record_retry(
        self,
        agent: str,
        reason: str,
        attempt: int,
    ) -> TraceEvent:
        """Convenience: record a retry."""
        return self.record(
            "retry",
            agent,
            f"retry attempt {attempt}",
            details={"reason": reason, "attempt": attempt},
        )

    # ----- metrics -----

    def total_tokens(self) -> int:
        return sum(e.tokens_used for e in self.events)

    def total_duration_ms(self) -> float:
        return (time.monotonic() - self.start_time) * 1000

    def model_usage(self) -> dict[str, int]:
        """Return ``{model_name: total_tokens}``."""
        usage: dict[str, int] = {}
        for e in self.events:
            if e.model:
                usage[e.model] = usage.get(e.model, 0) + e.tokens_used
        return usage

    def agent_timings(self) -> dict[str, float]:
        """Return ``{agent_name: total_ms}``."""
        timings: dict[str, float] = {}
        for e in self.events:
            timings[e.agent] = timings.get(e.agent, 0.0) + e.duration_ms
        return timings

    # ----- serialisation -----

    def finalize(self, **extra_summary: Any) -> dict[str, Any]:
        """Build the final summary and return the full trace as a dict."""
        self.summary = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "total_duration_ms": self.total_duration_ms(),
            "total_tokens": self.total_tokens(),
            "total_events": len(self.events),
            "model_usage": self.model_usage(),
            "agent_timings": self.agent_timings(),
            **extra_summary,
        }
        return {
            "summary": self.summary,
            "events": [e.to_dict() for e in self.events],
        }

    def save(self, directory: str | Path) -> Path:
        """Write the trace to ``<directory>/<run_id>.json``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"trace_{self.run_id}.json"
        data = self.finalize()
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_CURRENT_TRACE: ExecutionTrace | None = None


def start_trace(run_id: str | None = None) -> ExecutionTrace:
    """Start a new execution trace (replaces any previous one)."""
    global _CURRENT_TRACE
    _CURRENT_TRACE = ExecutionTrace(run_id=run_id)
    return _CURRENT_TRACE


def get_trace() -> ExecutionTrace | None:
    """Return the current trace, or ``None`` if none has been started."""
    return _CURRENT_TRACE
