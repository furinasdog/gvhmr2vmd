# GVHMR to MMD Bridge

[中文](README.zh-CN.md) · English · [中文完整手册](docs/USER_GUIDE.zh-CN.md) · [Full English Manual](docs/USER_GUIDE.en.md)

GVHMR to MMD Bridge converts world-grounded body motion from
[GVHMR](https://github.com/zju3dv/GVHMR) and MediaPipe hand landmarks into a safe NumPy
`.npz`, then retargets the motion to a standard MMD armature in Blender 4.5.

Highlights:

- optional body/trajectory smoothing, source clip trimming, and synchronized playback speed;
- in-place and fixed-position modes with retained previous Actions;
- world-space body motion, root translation, and timeline FPS;
- Japanese MMD bone names and mmd_tools `.L/.R` names;
- automatic MMD A-pose to SMPL T-pose upper-arm correction;
- 30 finger bones, per-frame confidence, and short-gap smoothing;
- reversible muting of conflicting foot IK constraints;
- pickle-free NPZ v2 and no PyTorch dependency inside Blender;
- a localhost-only Gradio WebUI suitable for SSH tunnelling.

Quick start:

1. Follow the [full English manual](docs/USER_GUIDE.en.md) to deploy the server integration.
2. Download `gvhmr_mmd_bridge.zip` from [Releases](https://github.com/furinasdog/gvhmr2vmd/releases), or run `make build` to build it in `dist/`.
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

Run the Blender regressions with your Blender 4.5 executable (repeat for each script):

```bash
blender --background --factory-startup --python-exit-code 1 --python tests/blender_processing.py
blender --background --factory-startup --python-exit-code 1 --python tests/blender_arm_retarget.py
blender --background --factory-startup --python-exit-code 1 --python tests/blender_finger_limits.py
blender --background --factory-startup --python-exit-code 1 --python tests/blender_heading_flip.py
blender --background --factory-startup --python-exit-code 1 --python tests/blender_shoulder_helpers.py
```

For an optional check against your own assets with mmd_tools installed:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python tests/blender_mmd_shoulder.py -- --pmx /path/model.pmx --motion /path/motion.npz
```

`make build` embeds the current Git commit and dirty state in the ZIP; the Blender panel displays
them below the status message. Packaging does not modify the source metadata file.

The unit tests also run in GitHub Actions on Python 3.10 and 3.12. Blender tests use synthetic
rigs; they do not replace visual checks with your own PMX and video. Local PMX files,
videos, checkpoints, inference outputs, and generated add-on archives are ignored by Git.

## Releases

The **Build and release** GitHub Actions workflow runs `make build`, verifies the
ZIP contents and Python syntax, and publishes the add-on plus `SHA256SUMS` to
GitHub Releases. It needs no custom token or server dependencies.

- Push a version tag such as `v0.2.0` to publish that commit automatically.
- Alternatively, open **Actions → Build and release → Run workflow**, select the
  branch or tag to build, and enter a release tag. A missing tag is created at
  the built commit; an existing tag must point to that same commit.
- Tags with a suffix such as `v0.3.0-beta.1` create prereleases. Published releases
  are never overwritten; a failed upload can be retried while the release is a draft.

## Project status

Current release: **0.2.0 (alpha)**. Body and finger retargeting have been tested with Blender
4.5 LTS and two standard miHoYo-style MMD skeleton layouts. Facial animation, cloth, hair,
physics baking, and multi-person tracking are outside the current scope.

## License

Copyright © 2026 GVHMR-to-MMD contributors. Project source: GPL-3.0-or-later.
Third-party software, models, weights, and user assets remain under their respective licenses.
