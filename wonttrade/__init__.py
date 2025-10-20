"""WontTrade package exports."""

from importlib import metadata


def get_version() -> str:
    """Return the installed package version."""
    try:
        return metadata.version("wonttrade")
    except metadata.PackageNotFoundError:  # pragma: no cover - fallback for local runs
        return "0.0.0"


__all__ = ["get_version"]
