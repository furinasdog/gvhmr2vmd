"""Packaging metadata tests that never modify the working repository."""

from __future__ import annotations

import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from tools.package_addon import (
    ADDON_NAME,
    BuildMetadata,
    build_archive,
    get_build_metadata,
    render_build_info,
)

ROOT = Path(__file__).resolve().parents[1]
BUILD_INFO_SOURCE = ROOT / "blender_addon" / ADDON_NAME / "build_info.py"
COMMIT = "1234567890abcdef1234567890abcdef12345678"


class GitMetadataTests(unittest.TestCase):
    def test_clean_git_checkout_records_full_and_short_commit(self):
        responses = [SimpleNamespace(stdout=f"{COMMIT}\n"), SimpleNamespace(stdout="")]
        with patch("tools.package_addon.subprocess.run", side_effect=responses) as git:
            metadata = get_build_metadata(Path("checkout"))
        self.assertEqual(metadata, BuildMetadata(COMMIT, COMMIT[:12], False))
        self.assertEqual(git.call_args_list[0].args[0][-3:], ["rev-parse", "--verify", "HEAD"])
        self.assertEqual(
            git.call_args_list[1].args[0][-3:],
            ["status", "--porcelain", "--untracked-files=normal"],
        )

    def test_modified_staged_or_untracked_files_mark_build_dirty(self):
        for status in (" M addon.py\n", "M  addon.py\n", "?? new_feature.py\n"):
            with self.subTest(status=status):
                responses = [SimpleNamespace(stdout=COMMIT), SimpleNamespace(stdout=status)]
                with patch("tools.package_addon.subprocess.run", side_effect=responses):
                    metadata = get_build_metadata(Path("checkout"))
                self.assertEqual(metadata.commit, COMMIT)
                self.assertIs(metadata.dirty, True)

    def test_missing_git_or_repository_falls_back_to_unknown(self):
        errors = (
            FileNotFoundError("git is unavailable"),
            subprocess.CalledProcessError(128, ["git"]),
            subprocess.TimeoutExpired(["git"], 10),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                with patch("tools.package_addon.subprocess.run", side_effect=error):
                    self.assertEqual(get_build_metadata(Path("source")), BuildMetadata())

    def test_invalid_git_hash_is_not_embedded(self):
        for value in ("HEAD", "1234567", "not a commit"):
            with self.subTest(value=value):
                with patch(
                    "tools.package_addon.subprocess.run",
                    return_value=SimpleNamespace(stdout=value),
                ):
                    self.assertEqual(get_build_metadata(Path("source")), BuildMetadata())

    def test_failed_status_does_not_claim_a_clean_checkout(self):
        responses = [
            SimpleNamespace(stdout=COMMIT),
            subprocess.CalledProcessError(128, ["git", "status"]),
        ]
        with patch("tools.package_addon.subprocess.run", side_effect=responses):
            metadata = get_build_metadata(Path("checkout"))
        self.assertEqual(metadata, BuildMetadata(COMMIT, COMMIT[:12], None))


class PackageTests(unittest.TestCase):
    def test_source_install_has_development_label(self):
        namespace = runpy.run_path(str(BUILD_INFO_SOURCE))
        self.assertEqual(namespace["GIT_COMMIT"], "unknown")
        self.assertIsNone(namespace["GIT_DIRTY"])
        self.assertEqual(namespace["build_label"](), "Build: development (unknown)")

    def test_archive_embeds_identity_and_keeps_source_files_unchanged(self):
        cases = (
            (BuildMetadata(COMMIT, COMMIT[:12], False), f"Build: {COMMIT[:12]}"),
            (BuildMetadata(COMMIT, COMMIT[:12], True), f"Build: {COMMIT[:12]} (dirty)"),
            (BuildMetadata(COMMIT, COMMIT[:12], None), f"Build: {COMMIT[:12]} (status unknown)"),
            (BuildMetadata(), "Build: development (unknown)"),
        )
        for metadata, label in cases:
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "blender_addon" / ADDON_NAME
                source.mkdir(parents=True)
                (source / "__init__.py").write_bytes(b'"""Test add-on."""\n')
                (source / "build_info.py").write_bytes(BUILD_INFO_SOURCE.read_bytes())
                (source / "feature.py").write_bytes(b"FEATURE = True\r\n")
                (root / "LICENSE").write_bytes(b"Test license\n")
                originals = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
                with patch("tools.package_addon.get_build_metadata", return_value=metadata):
                    archive_path = build_archive(root)
                self.assertEqual(archive_path, root / "dist" / f"{ADDON_NAME}.zip")
                for path, original in originals.items():
                    self.assertEqual(path.read_bytes(), original, f"Modified source: {path}")
                with ZipFile(archive_path) as archive:
                    self.assertIsNone(archive.testzip())
                    self.assertEqual(len(archive.namelist()), 4)
                    self.assertEqual(
                        set(archive.namelist()),
                        {
                            f"{ADDON_NAME}/{name}"
                            for name in ("__init__.py", "build_info.py", "feature.py", "LICENSE")
                        },
                    )
                    for name in ("__init__.py", "feature.py"):
                        self.assertEqual(
                            archive.read(f"{ADDON_NAME}/{name}"), originals[source / name]
                        )
                    self.assertEqual(
                        archive.read(f"{ADDON_NAME}/LICENSE"), originals[root / "LICENSE"]
                    )
                    content = archive.read(f"{ADDON_NAME}/build_info.py")
                namespace = {}
                exec(compile(content, "build_info.py", "exec"), namespace)
                self.assertEqual(namespace["GIT_COMMIT"], metadata.commit)
                self.assertEqual(namespace["GIT_COMMIT_SHORT"], metadata.short_commit)
                self.assertIs(namespace["GIT_DIRTY"], metadata.dirty)
                self.assertEqual(namespace["build_label"](), label)

    def test_malformed_metadata_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "GIT_COMMIT"):
            render_build_info("# Missing metadata constants\n", BuildMetadata())


if __name__ == "__main__":
    unittest.main()
