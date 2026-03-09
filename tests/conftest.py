"""Pytest configuration and fixtures."""

import sys
from unittest import mock

# Mock docker if it's not installed before importing issue_resolver modules
if 'docker' not in sys.modules:
    sys.modules['docker'] = mock.MagicMock()
