# Changelog

All notable changes are documented here. The project follows semantic versioning while it remains
possible for alpha releases to refine behavior.

## [Unreleased]

- GitHub-ready bilingual documentation, community files, CI, and license boundary clarification.
- Removed unofficial automatic download paths for SMPL and SMPL-X.

## [0.2.0] - 2026-09-29

- Added MediaPipe 21-point hand recognition using ViTPose-guided crops.
- Added NPZ v2 hand landmarks, confidence arrays, and backend metadata.
- Added retargeting for 30 MMD finger bones.
- Kept backward compatibility with body-only NPZ v1 files.
- Added automatic MMD A-pose to SMPL T-pose arm correction.
- Added Blender 4.5-safe IK state storage and restoration.

## [0.1.0] - 2026-09-29

- Initial safe NPZ exporter and Gradio WebUI.
- Initial Blender 4.5 body retargeter for standard Japanese MMD armatures.
- Added root translation scaling, FPS synchronization, direction flip, and IK muting.
