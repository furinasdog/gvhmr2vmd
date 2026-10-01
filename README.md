# GVHMR to MMD Bridge

[中文](README.zh-CN.md) · English · [中文完整手册](docs/USER_GUIDE.zh-CN.md) · [Full English Manual](docs/USER_GUIDE.en.md)

GVHMR to MMD Bridge converts world-grounded body motion from
[GVHMR](https://github.com/zju3dv/GVHMR) and MediaPipe hand landmarks into a safe NumPy
`.npz`, then retargets the motion to a standard MMD armature in Blender 4.5.

Highlights:

- world-space body motion, root translation, and timeline FPS;
- Japanese MMD bone names and mmd_tools `.L/.R` names;
- automatic MMD A-pose to SMPL T-pose upper-arm correction;
- 30 finger bones, per-frame confidence, and short-gap smoothing;
- reversible muting of conflicting foot IK constraints;
- pickle-free NPZ v2 and no PyTorch dependency inside Blender;
- a localhost-only Gradio WebUI suitable for SSH tunnelling.

Quick start:

1. Follow the [full English manual](docs/USER_GUIDE.en.md) to deploy the server integration.
2. Run `python tools/package_addon.py` to create `dist/gvhmr_mmd_bridge.zip`.
3. Install the ZIP in Blender 4.5 and import the PMX with mmd_tools.
4. Select the NPZ and armature in the “GVHMR MMD” panel, validate, and apply.

> [!IMPORTANT]
> This repository is GPL-3.0-or-later, while the upstream GVHMR license restricts use to
> educational, research, and non-profit purposes. SMPL, SMPL-X, MANO, model checkpoints,
> and PMX assets have separate terms and are not distributed here. See
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Repository layout

```text
blender_addon/gvhmr_mmd_bridge/  Blender 4.5 add-on source
server/                           GVHMR exporter, WebUI, and hand extraction
server/patches/                   Minimal optional GVHMR patch
docs/                             Motion format and bilingual manuals
tests/                            Unit and Blender integration tests
tools/                            PMX inspection and release packaging
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
ruff check .
python tools/package_addon.py
```

The Blender and real-PMX integration commands are documented in the manuals. Local PMX files,
videos, checkpoints, inference outputs, and generated add-on archives are ignored by Git.

## Project status

Current release: **0.2.0 (alpha)**. Body and finger retargeting have been tested with Blender
4.5 LTS and two standard miHoYo-style MMD skeleton layouts. Facial animation, cloth, hair,
physics baking, and multi-person tracking are outside the current scope.

## License

Copyright © 2026 GVHMR-to-MMD contributors. Project source: GPL-3.0-or-later.
Third-party software, models, weights, and user assets remain under their respective licenses.
