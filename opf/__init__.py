"""Public OPF package."""

__all__ = [
    "DecodeOptions",
    "INHERIT",
    "OPF",
    "RedactionResult",
    "redact",
    "redact_many",
]


def redact(text: str) -> str:
    """Redact one text string with the default local OPF model.

    Args:
        text: Input text to redact.

    Returns:
        The redacted text only.

    Raises:
        ValueError: If model configuration or decode settings are invalid.
        RuntimeError: If the local checkpoint cannot be loaded.
    """
    from ._api import redact as redact_impl

    return redact_impl(text)


def redact_many(
    texts: list[str] | tuple[str, ...],
    *,
    window_batch_size: int | None = None,
) -> tuple[str, ...]:
    """Redact multiple text strings with the default local OPF model."""
    from ._api import redact_many as redact_many_impl

    return redact_many_impl(texts, window_batch_size=window_batch_size)


def __getattr__(name: str):
    """Lazily expose the supported public API symbols from ``opf._api``."""
    if name == "DecodeOptions":
        from ._api import DecodeOptions

        return DecodeOptions
    if name == "INHERIT":
        from ._api import INHERIT

        return INHERIT
    if name == "OPF":
        from ._api import OPF

        return OPF
    if name == "RedactionResult":
        from ._api import RedactionResult

        return RedactionResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
