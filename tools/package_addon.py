"""Build the add-on ZIP with the current checkout's Git identity."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ADDON_NAME = "gvhmr_mmd_bridge"


@dataclass(frozen=True)
class BuildMetadata:
    commit: str = "unknown"
    short_commit: str = "unknown"
    dirty: bool | None = None


def get_build_metadata(root: Path) -> BuildMetadata:
    """Read Git without changing it; permit builds from source-only archives."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "--verify", "HEAD").lower()
    except (OSError, subprocess.SubprocessError):
        return BuildMetadata()
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
        return BuildMetadata()
    try:
        dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return BuildMetadata(commit, commit[:12], dirty)


def render_build_info(source: str, metadata: BuildMetadata) -> bytes:
    """Embed metadata while retaining the source module's label implementation."""
    values = {
        "GIT_COMMIT": metadata.commit,
        "GIT_COMMIT_SHORT": metadata.short_commit,
        "GIT_DIRTY": metadata.dirty,
    }
    for name, value in values.items():
        source, count = re.subn(rf"(?m)^{name} = .*?$", f"{name} = {value!r}", source)
        if count != 1:
            raise ValueError(f"Expected exactly one {name} constant in build_info.py")
    return source.encode("utf-8")


def build_archive(root: Path = ROOT, output: Path | None = None) -> Path:
    root = Path(root)
    source_root = root / "blender_addon" / ADDON_NAME
    output = Path(output) if output is not None else root / "dist" / f"{ADDON_NAME}.zip"
    sources = sorted(path for path in source_root.rglob("*.py") if "__pycache__" not in path.parts)
    for required in (source_root / "__init__.py", source_root / "build_info.py", root / "LICENSE"):
        if not required.is_file():
            raise FileNotFoundError(f"Missing package source: {required}")

    metadata = get_build_metadata(root)
    build_info = render_build_info(
        (source_root / "build_info.py").read_text(encoding="utf-8"), metadata
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for source in sources:
            name = f"{ADDON_NAME}/{source.relative_to(source_root).as_posix()}"
            if source == source_root / "build_info.py":
                archive.writestr(name, build_info)
            else:
                archive.write(source, name)
        archive.write(root / "LICENSE", f"{ADDON_NAME}/LICENSE")
    return output


def main():
    output = build_archive()
    print(f"Created {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
