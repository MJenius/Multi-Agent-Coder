"""Basic test suite for the issue resolver."""

import pytest


def test_imports():
    """Test that main modules can be imported."""
    from issue_resolver import config
    from issue_resolver.graph import app as agent_graph
    from issue_resolver.utils.github_utils import fetch_issue_details, submit_pull_request
    
    assert config is not None
    assert agent_graph is not None
    assert fetch_issue_details is not None
    assert submit_pull_request is not None


def test_config_values():
    """Test that configuration values are set."""
    from issue_resolver.config import (
        SANDBOX_WORKSPACE_DIR,
        OLLAMA_BASE_URL,
        RESEARCHER_MODEL,
    )
    
    assert SANDBOX_WORKSPACE_DIR is not None
    assert OLLAMA_BASE_URL is not None
    assert RESEARCHER_MODEL is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
