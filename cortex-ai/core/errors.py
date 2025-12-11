"""
Domain-specific exception hierarchy for the AI Folder Organizer.

All predictable, user-facing failures should raise subclasses of OrganizerError.
The CLI layer will catch OrganizerError and print friendly messages instead of
raw stack traces.
"""

class OrganizerError(Exception):
    """Base class for all known, user-facing errors in the organizer domain.

    Any exception that should result in a friendly CLI message (rather than
    a full stack trace) should inherit from this.
    """

# ---------------------------------------------------------------------------
# Path / filesystem related errors
# ---------------------------------------------------------------------------

class PathError(OrganizerError):
    """Base class for errors related to input paths and filesystem layout."""


class PathNotFoundError(PathError):
    """Raised when the provided root path does not exist.

    Example: user runs `organizer scan /not/a/real/path`.
    """


class PathNotDirectoryError(PathError):
    """Raised when the provided root path exists but is not a directory.

    Example: user passes a file instead of a folder as the root.
    """


class NoFilesFoundError(PathError):
    """Raised when the scan completes successfully but finds no files.

    The CLI may treat this as a soft error and print a friendly message like
    'No files found under ...' and exit with code 0 or 1 depending on policy.
    """


# ---------------------------------------------------------------------------
# Configuration / environment errors
# ---------------------------------------------------------------------------

class ConfigError(OrganizerError):
    """Raised when a configuration file is missing, malformed, or invalid.

    This covers issues in loading or validating a YAML/JSON config for
    categories, extension mappings, or thresholds.
    """


class EnvError(OrganizerError):
    """Base class for errors related to environment variables or .env files."""


class MissingApiKeyError(EnvError):
    """Raised when the LLM API key is not available in the environment.

    Example: OPENAI_API_KEY (or the chosen env var) is not set.
    """


# ---------------------------------------------------------------------------
# LLM / classification related errors
# ---------------------------------------------------------------------------

class LlmError(OrganizerError):
    """Base class for errors that occur while calling or using the LLM API."""


class LlmUnavailableError(LlmError):
    """Raised when the LLM service cannot be reached or returns a 5xx-level error.

    Examples:
    - Network timeout or connection error.
    - Provider returns a transient server error.
    """


class LlmResponseParseError(LlmError):
    """Raised when the LLM response cannot be parsed into the expected JSON format.

    Example: the model returns non-JSON or missing expected keys (category,
    confidence) and classification cannot proceed safely.
    """


class ClassificationError(OrganizerError):
    """Raised when classification fails in a way that is not just 'LLM unavailable'.

    Example: internal logic error while merging rule-based and LLM classifications,
    or invariant violations in the classification pipeline.
    """


# ---------------------------------------------------------------------------
# Execution / move / logging errors
# ---------------------------------------------------------------------------

class ExecutionError(OrganizerError):
    """Base class for errors that occur while applying an organization plan."""


class FileMoveError(ExecutionError):
    """Raised for catastrophic failures moving files (not per-file soft failures).

    Per-file failures (e.g., single permission error) are usually recorded in
    logs and reported in summary without aborting the whole run. This error is
    for cases like the root directory not being writable at all, or a critical
    invariant failing during move operations.
    """


class LoggingError(ExecutionError):
    """Raised when the tool fails to write or update its move log.

    Example: log directory is not writable or the CSV cannot be created.
    """
