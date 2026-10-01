from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
ADDON_NAME = "gvhmr_mmd_bridge"
SOURCE = ROOT / "blender_addon" / ADDON_NAME
OUTPUT = ROOT / "dist" / f"{ADDON_NAME}.zip"


def main():
    sources = sorted(
        path for path in SOURCE.rglob("*.py") if "__pycache__" not in path.parts
    )
    if not (SOURCE / "__init__.py").is_file():
        raise FileNotFoundError(f"Missing Blender add-on entry point: {SOURCE / '__init__.py'}")
    if not (ROOT / "LICENSE").is_file():
        raise FileNotFoundError(f"Missing project license: {ROOT / 'LICENSE'}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(source, f"{ADDON_NAME}/{source.relative_to(SOURCE).as_posix()}")
        archive.write(ROOT / "LICENSE", f"{ADDON_NAME}/LICENSE")

    print(f"Created {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
