"""CatBoost monthly tests are deferred until the direct multi-horizon rewrite."""

import pytest

pytest.skip(
    "CatBoost rewritten in Phase 2 (direct multi-horizon)",
    allow_module_level=True,
)
