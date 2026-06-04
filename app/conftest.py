"""Pytest configuration for the Streamlit app package.

Adds the app root to ``sys.path`` so tests can import the ``utils`` and ``ui``
packages the same way the Streamlit pages do (e.g. ``from utils.champion import
...``), regardless of the directory pytest is invoked from.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
