"""Build identity; source checkouts intentionally use development defaults."""

# tools/package_addon.py replaces these constants only inside the built ZIP.
GIT_COMMIT = "unknown"
GIT_COMMIT_SHORT = "unknown"
GIT_DIRTY = None


def build_label() -> str:
    """Return a compact build identity suitable for the Blender panel footer."""
    if GIT_COMMIT == "unknown":
        return "Build: development (unknown)"
    label = f"Build: {GIT_COMMIT_SHORT}"
    if GIT_DIRTY is True:
        return f"{label} (dirty)"
    if GIT_DIRTY is None:
        return f"{label} (status unknown)"
    return label
