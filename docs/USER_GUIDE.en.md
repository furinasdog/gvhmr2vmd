# GVHMR to MMD Bridge — Full User Manual

This manual covers server deployment, video inference, NPZ download, Blender 4.5 retargeting,
and optional VMD export. Replace every example path with an absolute path on your system.

## 1. Pipeline

```text
single-person video
  ├─ GVHMR: body pose, world-grounded root motion, camera motion
  ├─ ViTPose wrist/elbow cache → MediaPipe: 21 3D landmarks per hand
  └─ gvhmr_motion.npz (safe numeric arrays)
                         ↓
Blender 4.5 + mmd_tools → MMD armature → Action → optional VMD export
```

The add-on does not modify meshes, materials, or rigid bodies. Hair, clothes, and accessories
remain a physics or manual-animation task.

## 2. Requirements

### Inference server

- Linux;
- an NVIDIA CUDA GPU;
- Python 3.10;
- a working official GVHMR installation;
- legitimately obtained SMPL, SMPL-X, and GVHMR checkpoints;
- approximately 12 GB or more VRAM and 20 GB free disk space recommended.

### Animation workstation

- Blender 4.5 LTS;
- mmd_tools 4.x matching Blender 4.5;
- the add-on ZIP built from this repository;
- a PMX model whose terms permit your intended use.

ARP is not required. Direct FK retargeting is preferred for standard MMD skeletons; use an
additional rigging layer only for non-standard armatures.

## 3. Licensing and asset preparation

Read [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) first. In particular:

- upstream GVHMR permits educational, research, and non-profit use only;
- SMPL/SMPL-X require registration and acceptance of their own terms;
- do not commit PMX files, videos, checkpoints, SMPL files, inference NPZ files, or MANO files;
- this project never downloads SMPL/SMPL-X from unofficial mirrors.

Suggested layout:

```text
/work/GVHMR/                 official GVHMR checkout
/work/gvhmr-to-mmd-bridge/   this repository
```

## 4. Deploy the server integration

### 4.1 Install and verify GVHMR

Follow the upstream GVHMR instructions first. A typical start is:

```bash
git clone https://github.com/zju3dv/GVHMR.git /work/GVHMR
cd /work/GVHMR
conda create -n gvhmr python=3.10
conda activate gvhmr
python -m pip install -r requirements.txt
python -m pip install -e .
```

Place your licensed files at the paths expected upstream:

```text
inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
```

Install the remaining public checkpoints according to the upstream directory layout. Run the
official example successfully before adding this bridge; otherwise upstream and integration
problems are difficult to distinguish.

### 4.2 Copy the integration files

```bash
export BRIDGE_ROOT=/work/gvhmr-to-mmd-bridge
export GVHMR_ROOT=/work/GVHMR

cp "$BRIDGE_ROOT/server/gvhmr_export.py" "$GVHMR_ROOT/"
cp "$BRIDGE_ROOT/server/webui.py" "$GVHMR_ROOT/"
cp "$BRIDGE_ROOT/server/hand_pose_mediapipe.py" "$GVHMR_ROOT/"
git -C "$GVHMR_ROOT" apply "$BRIDGE_ROOT/server/patches/gvhmr_demo_skip_render.patch"
```

The patch only adds `--skip_render`; it does not alter the network or official `.pt` data. Skip
it if your upstream version already offers equivalent functionality.

### 4.3 Install WebUI and hand dependencies

Inside the GVHMR environment:

```bash
conda activate gvhmr
python -m pip install gradio==6.26.0
python -m pip install -r "$BRIDGE_ROOT/server/requirements-hand.txt"
python -m pip install --no-deps mediapipe==0.10.14
```

The pin prevents newer MediaPipe packages from upgrading NumPy/OpenCV and breaking the
Ultralytics version used by GVHMR. Verify the stack:

```bash
python - <<'PY'
import cv2, mediapipe, numpy, torch, ultralytics
print("numpy", numpy.__version__)
print("opencv", cv2.__version__)
print("mediapipe", mediapipe.__version__)
print("ultralytics", ultralytics.__version__)
print("cuda", torch.cuda.is_available())
PY
```

The tested combination is NumPy 1.23.5, OpenCV 4.11.0.86, MediaPipe 0.10.14. If your upstream
environment differs, preserve a working GVHMR stack and use a separate hand environment.

## 5. Start and access the WebUI

```bash
cd /work/GVHMR
conda activate gvhmr
python webui.py
```

The default bind is `127.0.0.1:7860`. Create a tunnel from your workstation:

```bash
ssh -L 7860:127.0.0.1:7860 -p <ssh-port> <user>@<server-host>
```

Open `http://127.0.0.1:7860` locally.

For an intentional non-local bind, authentication is mandatory:

```bash
export GVHMR_WEBUI_HOST=0.0.0.0
export GVHMR_WEBUI_PORT=7860
export GVHMR_WEBUI_USERNAME='your-user'
export GVHMR_WEBUI_PASSWORD='use-a-long-random-password'
python webui.py
```

The WebUI refuses an unauthenticated non-local bind. Never commit credentials.

## 6. Run inference

### Recommended input

- one person with the full body visible whenever possible;
- hands at least several dozen pixels wide and limited motion blur;
- enable “Static camera” for a locked camera;
- avoid unnecessary recompression;
- multi-person subject selection is not currently guaranteed.

### WebUI options

- **Static camera**: use for a locked camera;
- **Use DPVO**: optional, slower, and requires extra dependencies;
- **Focal length**: enter it when known, otherwise leave zero;
- **Export Blender/MMD NPZ**: normally enabled;
- **Render preview**: disable to skip expensive preview rendering;
- **Recognize hands and fingers**: crops hands from ViTPose wrists/elbows and runs the 21-point
  MediaPipe hand model.

The first hand run downloads Google's official `hand_landmarker.task`. Download these results:

- `gvhmr_motion.npz`: input for the Blender add-on;
- `hmr4d_results.pt`: raw GVHMR output for diagnostics only;
- preview video: produced only when rendering is enabled.

The NPZ contains numeric/string arrays only and is loaded with `allow_pickle=False`. See
[MOTION_FORMAT.md](MOTION_FORMAT.md).

## 7. Command-line export

Convert an existing official result:

```bash
cd /work/GVHMR
python gvhmr_export.py outputs/demo/example/hmr4d_results.pt \
  -o outputs/demo/example/gvhmr_motion.npz --fps 30
```

Add hand data to an existing NPZ:

```bash
python hand_pose_mediapipe.py \
  --video outputs/demo/example/0_input_video.mp4 \
  --vitpose outputs/demo/example/preprocess/vitpose.pt \
  --model inputs/checkpoints/mediapipe/hand_landmarker.task \
  --motion-npz outputs/demo/example/gvhmr_motion.npz
```

Use GVHMR's normalized `0_input_video.mp4` so frames align with the ViTPose cache.

## 8. Build and install the Blender add-on

From this repository:

```bash
python tools/package_addon.py
```

This creates `dist/gvhmr_mmd_bridge.zip`.

1. Open Blender 4.5.
2. Go to **Edit → Preferences → Add-ons**.
3. Choose **Install from Disk**.
4. Select the ZIP and enable **Animation: GVHMR to MMD Bridge**.
5. For an update, close Blender, overwrite the installation, and restart to clear module caches.

## 9. Import a model and apply motion

1. Import the PMX with mmd_tools.
2. Keep the armature in its original rest pose.
3. Press `N` in the 3D View and open **GVHMR MMD**.
4. Select `gvhmr_motion.npz`.
5. Select the model Armature, not a mesh or MMD root Empty.
6. Click **Validate file and bone mapping**.
7. Confirm the frame/bone counts, then click **Apply GVHMR motion**.

### Options

- **Start frame**: first target frame;
- **Use motion FPS**: synchronize the scene to the NPZ FPS;
- **Automatic root translation scale**: convert metres from skeleton height;
- **Flip forward 180°**: rotate the whole motion and translation path by half a turn around the armature Z axis, preserving relative joint motion. Reapply motion after changing this option. It does not remove time-varying heading drift in the source;
- **Correct MMD arm rest angle**: compensate MMD A-pose versus SMPL T-pose;
- **手指仅前后屈伸 (Finger flexion only)**: enabled by default; removes sideways motion and twist from all finger joints, including thumbs, while preserving the model's resting spread. Disable to retain full finger rotations. Reapply motion after changing this setting.
- **限制屈伸角度 (Limit finger angles)**: enabled by default within finger flexion mode. Each joint, including thumbs, is limited to 90° flexion and 10° extension relative to its rest pose. Adjust for your model; set maximum extension to 0° to prevent backward bending. Reapply motion to bake new limits. This does not detect finger collisions.
- **Apply finger motion**: drive 30 finger bones when hand data is present;
- **Hand confidence threshold**: default 0.35; raise for jitter, lower for missing keys;
- **Mute IK constraints**: prevent MMD foot IK from overriding FK leg motion.

Each application creates and assigns a new Action. Older Actions are retained for comparison and
rollback.

## 10. Cleanup and VMD export

- Inspect foot sliding, hand jitter, and occlusion segments in the Graph Editor.
- For long hand occlusions, remove bad keys and hold or animate the pose manually.
- Use **Restore IK constraints** to restore the original MMD constraint state.
- Bake physics only after the motion is final.
- Before exporting VMD with mmd_tools, verify the active Action, frame range, and bone names.

GVHMR does not provide facial expression data. The add-on does not create facial, lip-sync, hair,
or skirt animation.

## 11. Troubleshooting

### Missing `腕.L`, `足首.R`, or similar bones

Use the latest add-on. mmd_tools converts names such as `左腕` to Blender `.L/.R` names; version
0.2.0 supports both. Also confirm that the selected object is the Armature.

### Arms are globally too low or high

Keep arm-rest correction enabled for a standard MMD A-pose. Disable it for a true T-pose model.

### Fingers do not move

Confirm the NPZ is v2, the WebUI printed hand detection rates, and finger motion is enabled.
Very small or motion-blurred hands reduce detection quality.

### Fingers jitter or flip briefly

Raise the confidence threshold and delete isolated bad keys in the Graph Editor. Occlusion and
crossed hands remain limitations of the lightweight single-person backend.

### Root sliding or wrong scale

Check the PMX import scale. Start with automatic scale; disable it and tune the manual factor only
when necessary.

### WebUI does not start

Run `python tools/webui_smoke_test.py`, verify checkpoints, Gradio, and the port. A public bind
requires authentication.

### MediaPipe upgraded NumPy

Restore the NumPy/OpenCV versions required by GVHMR and reinstall the pinned MediaPipe with
`--no-deps` as shown in section 4.3.
