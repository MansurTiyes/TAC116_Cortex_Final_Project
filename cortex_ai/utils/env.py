"""
Environment helper utilities for the AI Folder Organizer.

Responsible for:
- Loading environment variables from a .env file.
- Providing helpers to access the LLM API key in a safe, centralized way.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..core.errors import MissingApiKeyError

# Name of the env var that will store the LLM API key.
LLM_API_KEY_ENV_VAR = "OPENAI_API_KEY"

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    # Load from a .env in the current working directory or its parents.
    # This is called once at import time.
    load_dotenv()
    DOTENV_LOADED = True
except ImportError:  # pragma: no cover - only hit if python-dotenv isn't installed
    DOTENV_LOADED = False

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_llm_api_key() -> str:
    """
    Return the LLM API key from the environment.

    - Reads the key from the LLM_API_KEY_ENV_VAR (e.g. OPENAI_API_KEY).
    - Raises MissingApiKeyError if the key is not set.

    This should be the *only* place in the codebase that knows the actual env
    var name, so that if we ever rename it, we only change it here.
    """
    key = os.getenv(LLM_API_KEY_ENV_VAR)

    if not key:
        raise MissingApiKeyError(
            f"{LLM_API_KEY_ENV_VAR} not set. "
            "Please set it in your environment or in a .env file."
        )

    return key


def is_llm_key_present() -> bool:
    """
    Return True if the LLM API key appears to be set in the environment.

    Used by things like `organizer version` to show a quick status without
    raising an error.
    """
    return bool(os.getenv(LLM_API_KEY_ENV_VAR))