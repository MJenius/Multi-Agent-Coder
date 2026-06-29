"""Candidate Generator Node — generates multiple implementation patches.

Generates patches from multiple models in parallel to offer alternative perspectives
and improve likelihood of correctness.
"""

from __future__ import annotations

import concurrent.futures
from langchain_core.messages import SystemMessage, HumanMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.llm_utils import invoke_with_role_fallback
from issue_resolver.core.prompt_registry import get_prompt_registry

# Register default prompt on import
_DEFAULT_PROMPT = """\
You are an expert software engineer. Given a structured plan and code context, write a search/replace patch block.

OUTPUT FORMAT:
Output ONLY search/replace blocks in this format:
<<<<<<< SEARCH
[exact lines to find]
=======
[replacement lines]
>>>>>>> REPLACE

Critical Rules:
1. Make sure SEARCH blocks exactly match the spacing, indentation, and content in the original code.
2. Only modify the files mentioned in the plan.
3. Keep changes as minimal as possible to resolve the issue.
4. Output no conversation, explanation, or markdown wrappers outside the SEARCH/REPLACE blocks.
"""

get_prompt_registry().register("candidate_generator", "1.0", _DEFAULT_PROMPT)


def _generate_candidate_for_model(
    model_role: str,
    default_model: str,
    messages: list[any],
    context: dict,
) -> dict:
    """Helper invoked in a background thread to generate one candidate."""
    try:
        resp, chosen_model = invoke_with_role_fallback(
            role=model_role,
            candidates=[default_model],
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            context=context,
        )
        return {
            "success": True,
            "model": chosen_model,
            "patch": getattr(resp, "content", "").strip(),
        }
    except Exception as exc:
        print(f"[CandidateGenerator] [ERROR] Model {default_model} failed: {exc}")
        return {
            "success": False,
            "model": default_model,
            "error": str(exc),
        }


def candidate_generator_node(state: AgentState) -> dict:
    """Generate multiple candidate fixes in parallel."""
    print("[CandidateGenerator] Generating patch candidates in parallel...")
    issue = state.get("issue", "")
    file_context = state.get("file_context", [])
    structured_plan = state.get("structured_plan", {})

    prompt = get_prompt_registry().get("candidate_generator")

    context_str = "\n\n".join(file_context[:4]) if file_context else "(no context)"

    prompt_content = f"""Issue Description:
{issue}

## Plan:
{state.get("plan", "")}

## Context:
{context_str}

Generate the SEARCH/REPLACE block.
"""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=prompt_content),
    ]

    # Models to run in parallel
    # 1. Primary implementation (GPT-OSS 120B)
    # 2. Bugfix retry (DeepSeek v4 Flash)
    # 3. Code analyst (Qwen3.5 122B)
    candidates_to_run = [
        ("implementation", "openai/gpt-oss-120b"),
        ("bugfix_retry", "deepseek-ai/deepseek-v4-flash"),
        ("repo_analyst", "qwen/qwen3.5-122b-a10b"),
    ]

    results = []
    # Run in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                _generate_candidate_for_model,
                role,
                model_name,
                messages,
                {"issue_category": state.get("issue_category", "Bug")},
            ): model_name
            for role, model_name in candidates_to_run
        }

        for fut in concurrent.futures.as_completed(futures):
            model_name = futures[fut]
            try:
                res = fut.result()
                if res["success"]:
                    results.append(res)
                    print(f"[CandidateGenerator] Candidate from {res['model']} successfully generated")
            except Exception as exc:
                print(f"[CandidateGenerator] Thread execution for {model_name} failed: {exc}")

    # Fallback: if all parallel runs failed, try invoke_with_role_fallback synchronously on default coder candidate
    if not results:
        print("[CandidateGenerator] [WARNING] Parallel generation failed. Running fallback synchronously...")
        try:
            resp, chosen_model = invoke_with_role_fallback(
                role="implementation",
                candidates=["openai/gpt-oss-120b", "meta/llama-3.3-70b-instruct"],
                messages=messages,
                temperature=0.5,
                max_tokens=4096,
            )
            results.append({
                "success": True,
                "model": chosen_model,
                "patch": getattr(resp, "content", "").strip(),
            })
        except Exception as exc:
            print(f"[CandidateGenerator] Fallback implementation failed: {exc}")

    # Record trace
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "candidates_generated",
            "CandidateGenerator",
            f"Generated {len(results)} patch candidates",
            details={"models": [res["model"] for res in results]},
        )

    return {
        "candidate_patches": results,
        "history": append_to_history(
            "CandidateGenerator",
            "Generate",
            f"Generated {len(results)} patch candidates from different models.",
        ),
    }
