"""Classifier node for categorising incoming GitHub issues.

Classifies issues into categories: Bug, Performance, Security,
Documentation, Testing, Feature, API Change, Dependency Update,
Refactor, Typing, Configuration.

Uses a fast deterministic classifier first. Falls back to an LLM
only when the deterministic confidence is low (< 0.6).
"""

from __future__ import annotations

import re
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage

from issue_resolver.state import AgentState
from issue_resolver.utils.logger import append_to_history
from issue_resolver.llm_utils import invoke_with_role_fallback
from issue_resolver.core.prompt_registry import get_prompt_registry


# Register prompt template on import
_DEFAULT_PROMPT = """\
You are an expert AI triage agent. Your job is to classify the user's issue description into exactly ONE of the following categories:

Categories:
- Bug (functional error, unexpected behavior, crash)
- Performance (slowness, memory leak, high CPU)
- Security (vulnerability, credentials exposure)
- Documentation (missing docs, typos in readme, comments)
- Testing (adding tests, broken test suite)
- Feature (request for new functionality)
- API Change (breaking interface changes, endpoint updates)
- Dependency Update (updating packages, changing lockfiles)
- Refactor (improving code structure, cleanups)
- Typing (adding type hints, fixing type checks)
- Configuration (updating setting files like .env or config files)

Output ONLY the category name from the list above. Do not output anything else.
"""

get_prompt_registry().register("issue_classifier", "1.0", _DEFAULT_PROMPT)


def clean_issue_for_classification(issue_text: str) -> str:
    """Strip GitHub issue templates, HTML comments, checklists, boilerplate, and links."""
    if not issue_text:
        return ""
    
    # 1. Remove HTML comments
    text = re.sub(r'<!--[\s\S]*?-->', '', issue_text)
    
    # 2. Split into lines
    lines = text.split('\n')
    cleaned_lines = []
    
    # Common boilerplate sentences/phrases to remove
    boilerplate_patterns = [
        r'before (creating|opening|submitting) this issue',
        r'please (make sure|ensure|read|check|confirm|fill|review)',
        r'search existing issues',
        r'latest version',
        r'contribution guide',
        r'code of conduct',
        r'security policy',
        r'tick the boxes',
        r'replace this text',
        r'delete this section',
        r'fill out the template',
    ]
    
    # 3. Process line by line
    for line in lines:
        # Strip checklist boxes (e.g. - [ ] or - [x])
        if re.match(r'^\s*[-*+]\s*\[[ xX]\]', line):
            # Strip the checkbox prefix
            content = re.sub(r'^\s*[-*+]\s*\[[ xX]\]\s*', '', line)
            # Check if the content is boilerplate
            is_boilerplate = False
            for pattern in boilerplate_patterns:
                if re.search(pattern, content.lower()):
                    is_boilerplate = True
                    break
            if not is_boilerplate and content.strip():
                cleaned_lines.append(content)
            continue
        
        # Skip header-only lines that are typical boilerplate headers
        lower_line = line.strip().lower()
        boilerplate_headers = [
            "### checklist", "## checklist", "# checklist", "checklist:",
            "### code of conduct", "## code of conduct", "code of conduct:",
            "### prerequisites", "## prerequisites", "prerequisites:",
            "### guidelines", "## guidelines", "guidelines:",
        ]
        if any(header == lower_line or lower_line.startswith(header) for header in boilerplate_headers):
            continue
            
        # General boilerplate sentence check
        is_boilerplate = False
        for pattern in boilerplate_patterns:
            if re.search(pattern, lower_line):
                is_boilerplate = True
                break
        if is_boilerplate:
            continue
            
        cleaned_lines.append(line)
        
    text = '\n'.join(cleaned_lines)
    
    # 4. Remove documentation and other links
    def repl_markdown_link(match):
        link_text = match.group(1)
        url = match.group(2)
        url_lower = url.lower()
        text_lower = link_text.lower()
        doc_indicators = ["doc", "wiki", "guide", "manual", "readme", "faq", "contributing", "github.com/"]
        if any(ind in url_lower or ind in text_lower for ind in doc_indicators):
            return ""
        return link_text
        
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)', repl_markdown_link, text)
    text = re.sub(r'https?://[^\s]+', '', text)
    
    # Strip duplicate empty lines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()


def _deterministic_classify(issue_text: str) -> tuple[str, dict[str, Any]]:
    """Determine category and compute suitability metadata using heuristics."""
    issue_lower = issue_text.lower()
    
    scores = {
        "Bug": 0.0,
        "Performance": 0.0,
        "Security": 0.0,
        "Documentation": 0.0,
        "Testing": 0.0,
        "Feature": 0.0,
        "API Change": 0.0,
        "Dependency Update": 0.0,
        "Refactor": 0.0,
        "Typing": 0.0,
        "Configuration": 0.0
    }
    
    # Bug keywords
    bug_kws = ["bug", "crash", "error", "exception", "fail", "unexpected", "incorrect", 
               "wrong", "issue", "defect", "broken", "npe", "nullpointer", "stacktrace", 
               "traceback", "typeerror", "valueerror", "keyerror", "runtimeerror"]
    for kw in bug_kws:
        if kw in issue_lower:
            scores["Bug"] += 1.5
    if re.search(r"traceback\s*\(most\s+recent\s+call\s+last\)", issue_lower):
        scores["Bug"] += 5.0
    if re.search(r"\w+error:", issue_lower):
        scores["Bug"] += 3.0
    if re.search(r"caused by:", issue_lower):
        scores["Bug"] += 2.0
        
    # Performance keywords
    perf_kws = ["slow", "performance", "memory leak", "leak", "latency", "hang", "timeout", 
                "speed", "fast", "cpu", "profiling", "benchmark", "optimize", "optimization", 
                "out of memory", "oom"]
    for kw in perf_kws:
        if kw in issue_lower:
            scores["Performance"] += 1.5
            
    # Security keywords
    sec_kws = ["security", "vulnerability", "cve", "exploit", "leak credentials", "secret", 
               "password", "token", "auth", "xss", "csrf", "injection", "key", "cert", "ssl", "tls"]
    for kw in sec_kws:
        if kw in issue_lower:
            scores["Security"] += 1.5
            
    # Documentation keywords
    doc_kws = ["readme", "doc", "documentation", "docstring", "comment", "typo", "spell", 
               "grammar", "wiki", "manual", "guide", "tutorial"]
    for kw in doc_kws:
        if kw in issue_lower:
            scores["Documentation"] += 1.5
    if re.search(r"\b[a-zA-Z0-9_\-\./]+\.md\b", issue_lower):
        scores["Documentation"] += 3.0
        
    # Testing keywords
    test_kws = ["test", "pytest", "unittest", "coverage", "fixture", "mock", "assert", 
                "suite", "runner", "ci", "jenkins", "github actions", "workflow"]
    for kw in test_kws:
        if kw in issue_lower:
            scores["Testing"] += 1.5
            
    # Feature keywords
    feat_kws = ["feature", "enhance", "add support", "support for", "implement", "request", 
                "new functionality", "add option", "capability", "proposal", "wishlist"]
    for kw in feat_kws:
        if kw in issue_lower:
            scores["Feature"] += 1.5
            
    # API Change keywords
    api_kws = ["api", "endpoint", "breaking change", "deprecate", "http", "rest", "graphql", 
               "interface", "parameter", "signature", "client", "sdk", "compatibility"]
    for kw in api_kws:
        if kw in issue_lower:
            scores["API Change"] += 1.5
            
    # Dependency Update keywords
    dep_kws = ["dependency", "dependencies", "upgrade", "bump", "package", "version", "pip", 
               "npm", "requirements.txt", "package.json", "lockfile", "poetry", "yarn"]
    for kw in dep_kws:
        if kw in issue_lower:
            scores["Dependency Update"] += 1.5
            
    # Refactor keywords
    ref_kws = ["refactor", "cleanup", "reorganize", "structure", "architecture", "lint", 
               "formatting", "style", "dead code", "unused", "simplify", "rewrite"]
    for kw in ref_kws:
        if kw in issue_lower:
            scores["Refactor"] += 1.5
            
    # Typing keywords
    type_kws = ["type hint", "typing", "mypy", "pyright", "type check", "annotation", 
                "types", "generics", "protocol"]
    for kw in type_kws:
        if kw in issue_lower:
            scores["Typing"] += 1.5
            
    # Configuration keywords
    conf_kws = [".env", "config", "configuration", "settings", "settings.py", "pyproject.toml", 
                "setup.cfg", "dotenv", "environment variable", "json config", "yaml", "yml", "ini"]
    for kw in conf_kws:
        if kw in issue_lower:
            scores["Configuration"] += 1.5
            
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_cat, best_score = sorted_scores[0]
    second_cat, second_score = sorted_scores[1]
    
    if best_score == 0:
        confidence = 0.0
    else:
        margin = best_score - second_score
        confidence = min(1.0, (best_score / 4.0) * (1.0 + margin / (best_score + 0.1)))
        confidence = max(0.1, confidence)
        
    repro = "medium"
    if "traceback" in issue_lower or "error" in issue_lower or "```" in issue_lower:
        repro = "high"
    elif "intermittent" in issue_lower or "flaky" in issue_lower or "sometimes" in issue_lower:
        repro = "low"
        
    complexity = "medium"
    if len(issue_text) < 300 and not "traceback" in issue_lower:
        complexity = "low"
    elif len(issue_text) > 2000 or "architect" in issue_lower or "design" in issue_lower or "rewrite" in issue_lower:
        complexity = "high"
        
    verif = "runtime tests"
    if best_cat == "Typing":
        verif = "static type checking"
    elif best_cat == "Documentation":
        verif = "documentation validation"
    elif best_cat == "Configuration":
        verif = "configuration verification"
    elif best_cat == "Refactor":
        verif = "linting"
    elif best_cat == "Performance":
        verif = "performance benchmarking"
        
    arch_impact = "local"
    if complexity == "high":
        arch_impact = "system"
    elif complexity == "medium":
        arch_impact = "module"
        
    suitability = {
        "reproducibility": repro,
        "complexity": complexity,
        "verification_method": verif,
        "architectural_impact": arch_impact,
        "confidence": round(confidence, 2)
    }
    
    return best_cat, suitability


def issue_classifier_node(state: AgentState) -> dict:
    """Classify the incoming issue category (deterministic with LLM fallback)."""
    print("[Classifier] Categorising issue...")
    issue = state.get("issue", "")
    
    cleaned_issue = clean_issue_for_classification(issue)

    # Run deterministic classification first
    category, suitability = _deterministic_classify(cleaned_issue)
    method = "deterministic"

    # Fall back to LLM if confidence is low
    if suitability["confidence"] < 0.6:
        print(f"[Classifier] Deterministic confidence ({suitability['confidence']}) is low. Falling back to LLM...")
        prompt = get_prompt_registry().get("issue_classifier")

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Issue Description:\n{cleaned_issue}"),
        ]

        try:
            response, model_name = invoke_with_role_fallback(
                role="issue_classifier",
                candidates=["meta/llama-3.3-70b-instruct"],
                messages=messages,
                temperature=0.2,
                max_tokens=128,
                context={"issue_length": len(issue)},
            )
            output = response.content.strip()
            method = "llm_fallback"

            valid_categories = {
                "Bug", "Performance", "Security", "Documentation", "Testing",
                "Feature", "API Change", "Dependency Update", "Refactor", "Typing",
                "Configuration"
            }

            for cat in valid_categories:
                if cat.lower() in output.lower():
                    category = cat
                    break
                    
            print(f"[Classifier] LLM classified category as: {category} (using {model_name})")
            suitability["confidence"] = 0.8  # Default LLM confidence
            
        except Exception as e:
            print(f"[Classifier] LLM fallback failed: {e}. Using deterministic category: {category}")
            method = "deterministic_error_fallback"
    else:
        print(f"[Classifier] Deterministic match with confidence {suitability['confidence']}: {category}")

    # Record trace event if running
    from issue_resolver.core.execution_trace import get_trace
    trace = get_trace()
    if trace:
        trace.record(
            "issue_classified",
            "Classifier",
            f"classified as {category} ({method})",
            details={
                "method": method, 
                "category": category,
                "suitability": suitability
            },
        )

    return {
        "issue_category": category,
        "classification_method": method,
        "issue_suitability": suitability,
        "history": append_to_history(
            "Classifier",
            "Classify",
            f"Classified issue as {category} ({method}, confidence {suitability['confidence']})",
        ),
    }
