"""Candidate Evaluator Node — performs multi-objective patch scoring.

Scores patches along six dimensions: Correctness, Maintainability, Simplicity,
Style Consistency, Performance, and Risk, selecting the highest-weighted option.
"""

from __future__ import annotations

import json
import re
from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.llm_utils import invoke_with_role_fallback
from issue_resolver.core.prompt_registry import get_prompt_registry

# Register prompt template on import
_DEFAULT_PROMPT = """\
You are an expert code reviewer. You are given a bug description, code context, and a proposed code patch.
Evaluate this patch along the following seven dimensions, giving each a score from 0.0 to 10.0:

1. Correctness: Does the patch resolve the core issue without causing new failures?
2. Maintainability: Is the code clean, readable, and structured nicely?
3. Simplicity: Are changes minimal and straightforward, avoiding over-engineering?
4. Style Consistency: Does the code adhere to the rest of the repository's conventions?
5. Performance: Does the change avoid causing any performance regressions?
6. Risk: How likely is this to break other systems (high score means low risk, i.e. 10.0 = completely safe, 0.0 = highly risky)?
7. Repository Consistency: Does it follow the repository naming conventions, import styles, design patterns, and framework integration patterns as detailed in the repository context?

You must output a single JSON block wrapped in a ```json markdown block:
{
  "scores": {
    "correctness": 0.0,
    "maintainability": 0.0,
    "simplicity": 0.0,
    "style_consistency": 0.0,
    "performance": 0.0,
    "risk": 0.0,
    "repository_consistency": 0.0
  },
  "rationale": "Brief rationale for scores"
}
"""

get_prompt_registry().register("candidate_evaluator", "1.0", _DEFAULT_PROMPT)

# Evaluation weights
_WEIGHTS = {
    "correctness": 0.30,
    "maintainability": 0.15,
    "simplicity": 0.15,
    "style_consistency": 0.10,
    "performance": 0.10,
    "risk": 0.10,
    "repository_consistency": 0.10,
}


def candidate_evaluator_node(state: AgentState) -> dict:
    """Evaluate and select the best candidate patch using multi-objective scoring."""
    candidates = state.get("candidate_patches", [])
    if not candidates:
        print("[CandidateEvaluator] [WARNING] No candidates found to evaluate")
        return {"proposed_fix": ""}

    issue = state.get("issue", "")
    prompt = get_prompt_registry().get("candidate_evaluator")

    evaluated_candidates = []
    print(f"[CandidateEvaluator] Evaluating {len(candidates)} candidates using multi-objective weights...")

    for i, candidate in enumerate(candidates):
        model_name = candidate["model"]
        patch = candidate["patch"]

        eval_content = f"""Issue: {issue}
Proposed Patch from model '{model_name}':
{patch}
"""

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=eval_content),
        ]

        try:
            resp, chosen_model = invoke_with_role_fallback(
                role="reviewer",
                candidates=["nvidia/nemotron-3-ultra-550b-a55b"],
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
            raw = getattr(resp, "content", "") or ""
            m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
            json_str = m.group(1).strip() if m else raw.strip()
            eval_data = json.loads(json_str)

            # Calculate composite score
            scores = eval_data.get("scores", {})
            composite_score = sum(
                float(scores.get(dim, 5.0)) * _WEIGHTS[dim]
                for dim in _WEIGHTS
            )

            evaluated_candidates.append({
                "model": model_name,
                "patch": patch,
                "scores": scores,
                "composite_score": composite_score,
                "rationale": eval_data.get("rationale", ""),
            })
            print(f"[CandidateEvaluator] Candidate '{model_name}' scored: {composite_score:.2f}/10")
        except Exception as exc:
            print(f"[CandidateEvaluator] [WARNING] Failed to evaluate candidate '{model_name}': {exc}")
            # Give a default low score so we don't crash
            evaluated_candidates.append({
                "model": model_name,
                "patch": patch,
                "scores": {},
                "composite_score": 1.0,
                "rationale": f"Evaluation failed: {exc}",
            })

    # Sort candidates by composite score descending
    evaluated_candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    best_candidate = evaluated_candidates[0]

    print(
        f"[CandidateEvaluator] Selected patch from model '{best_candidate['model']}' "
        f"with score {best_candidate['composite_score']:.2f}"
    )

    # Record trace
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "candidates_evaluated",
            "CandidateEvaluator",
            f"Evaluated candidates, selected {best_candidate['model']}",
            details={
                "selected_model": best_candidate["model"],
                "score": best_candidate["composite_score"],
                "scores_breakdown": best_candidate["scores"],
                "all_candidates": [
                    {"model": c["model"], "score": c["composite_score"]}
                    for c in evaluated_candidates
                ]
            },
        )

    return {
        "proposed_fix": best_candidate["patch"],
        "candidate_scores": evaluated_candidates,
        "history": append_to_history(
            "CandidateEvaluator",
            "Select",
            f"Selected patch from {best_candidate['model']} (score {best_candidate['composite_score']:.2f}/10).",
        ),
    }
