# Phase 0 — Environment Setup: Results

**Status:** complete, all checks passing
**Date:** 2026-09-15
**Platform:** Windows 11 (26200) · Python 3.12.4 · RTX 4050 Laptop 6 GB · driver 581.86 · CUDA 13.0

---

## Summary

The stack runs natively on Windows. WSL is not required.

Getting there needed four workarounds and one version downgrade, all because
robosuite and robomimic support Linux and macOS only.

The finding that matters most is not a workaround. **On this Optimus laptop,
robosuite renders on the Intel integrated GPU unless a CUDA context is created
first.** That mistake costs 6.4× in per-step wall clock and far worse tail latency,
and it is completely silent — the images come back correct either way.

Once the correct GPU is used, rendering is close to free and physics dominates the
step. The premise in the phase documents that rendering is the expensive part does
not hold on this machine.

---

## Verification

`python scripts/check_env.py` — all 19 checks pass.

```
  [PASS] python                         3.12.4 on Windows
  [PASS] import torch                   2.11.0+cu128
  [PASS] import mujoco                  3.1.6
  [PASS] import robosuite               1.4.1
  [PASS] import robomimic               0.3.0
  [PASS] import lerobot                 0.6.1
  [PASS] import numpy                   2.2.6
  [PASS] cuda available                 torch 2.11.0+cu128
  [PASS] gpu device                     NVIDIA GeForce RTX 4050 Laptop GPU
  [PASS] gpu vram                       6.00 GiB total, 4.95 GiB free (budget 5.0 GiB)
  [PASS] cuda runtime                   cuda 12.8, capability 8.9
  [PASS] mujoco_gl backend              wgl
  [PASS] build env                      Lift/Panda/OSC_POSE, action_dim=7
  [PASS] render gpu                     NVIDIA GeForce RTX 4050 Laptop GPU/PCIe/SSE2
  [PASS] camera robot0_eye_in_hand      shape=(128, 128, 3) dtype=uint8
  [PASS] camera agentview               shape=(128, 128, 3) dtype=uint8
  [PASS] proprio state                  dim=32
  [PASS] render vram                    0.36 GiB consumed by the render context
  [PASS] per-step cost (2 cam @ 128px)  mean=7.96ms p95=10.46ms
```

---

## The GPU selection finding

robosuite's OpenGL context follows whichever GPU the driver believes the process
needs. A process that has not touched CUDA gets the Intel iGPU. Measured back to
back, two runs each, Lift with both cameras at 128 px:

| Process does | OpenGL renderer | mean | p95 |
| --- | --- | --- | --- |
| robosuite only | Intel RaptorLake-S Mobile Graphics | 46.95 ms | 54.46 ms |
| **CUDA init, then robosuite** | **NVIDIA RTX 4050** | **7.33 ms** | **8.71 ms** |

**6.4× on the mean, 6.3× on p95, reproducible across repeated runs.**

This is enforced in code, not left to import order:
[`prefer_discrete_gpu()`](../../src/envs/robosuite_compat.py) creates a one-element
CUDA tensor and is called from `configure_rendering()` before robosuite is
imported. Every script that builds an environment must go through that function.

### How this was nearly missed

The first version of `check_env.py` queried `GL_RENDERER` from a **separate GLFW
probe context** and reported `NVIDIA GeForce RTX 4050` — a PASS. The reading was
real but answered the wrong question: the probe context and robosuite's context can
land on different GPUs. Querying from inside the live robosuite context returned
`Intel`.

The symptom that exposed it was `nvidia-smi` reporting 0% utilisation and a 210 MHz
clock during 600 steps of sustained rendering. The check now queries inside
robosuite's own context and emits a `WARN` row, with the fix, when the renderer is
not the discrete GPU.

---

## Measurements

### Per-step wall clock — `Lift` / Panda / OSC_POSE, discrete GPU

50 warmup steps discarded, 300 steps timed, one process per configuration.

| Configuration | Resolution | Mean | p95 | Min |
| --- | --- | --- | --- | --- |
| physics only | — | 5.92 ms | 6.80 ms | 5.14 ms |
| `agentview` | 128 | 6.90 ms | 7.89 ms | 6.01 ms |
| `robot0_eye_in_hand` | 128 | 6.88 ms | 7.55 ms | 5.90 ms |
| **both cameras** | **128** | **7.51 ms** | **8.79 ms** | **6.47 ms** |
| both cameras | 256 | 7.38 ms | 8.22 ms | 6.46 ms |
| both cameras | 512 | 8.05 ms | 9.01 ms | 6.81 ms |

Two conclusions, both of which revise assumptions in the phase documents:

1. **The second camera is cheap.** Roughly 0.6 ms, about 9% of the step. Keeping
   `agentview` for debuggability costs almost nothing. `phase-0-setup.md` expects
   two views to roughly double the render cost; here rendering is not the
   bottleneck at all.
2. **Resolution is nearly free.** 128 px to 512 px costs 0.5 ms. `phase-3-dataset.md`
   §3.2 advises against rendering high and downscaling on cost grounds; on this
   machine that cost does not exist. Render at whatever Phase 1 determines SmolVLA
   expects, and choose it on correctness grounds alone.

Physics at 5.92 ms is the floor and is CPU-bound.

### Runtime projections

At 7.51 ms/step with both cameras, excluding model inference:

| Workload | Estimate |
| --- | --- |
| One 200-step episode | ~1.5 s |
| Phase 2 zero-shot, 50 episodes | ~75 s |
| Phase 3 regeneration, Lift-PH (200 demos × ~55 steps) | ~1.4 min |

Neither heavy phase is rendering-bound. Phase 2 will be dominated by SmolVLA
inference latency, which Phase 1 measures.

### VRAM

The render context consumes **0.36 GiB**. Against the 5.0 GiB budget this leaves
ample room, but it is not free and must be counted in Phase 4 and Phase 5 planning
alongside model weights and optimizer state.

---

## Deviations from `phase-0-setup.md`

### §0.3 — `MUJOCO_GL=egl` does not apply

The document prescribes EGL with `osmesa` as fallback. **EGL does not exist on
Windows.** robosuite's validator accepts only `wgl` and `glfw` here, and `osmesa`
is Linux-only in its backend dispatch (`robosuite/utils/binding_utils.py:46-60`).

Working configuration: **`MUJOCO_GL=wgl`**, resolving to `GLFWGLContext`. This is
GPU rendering — the `osmesa` CPU-rendering penalty the document warns about does
not apply.

### Install order changed

The document puts torch first. Actual order was MuJoCo/robosuite → render test →
torch → robomimic → lerobot, so the cheap step most likely to fail (~50 MB) ran
before the expensive one (~2.5 GB). The render test failed three times before
working, so the ordering paid for itself.

---

## Workarounds

All four live in [`src/envs/robosuite_compat.py`](../../src/envs/robosuite_compat.py),
are idempotent, and are no-ops on Linux and macOS. They patch installed packages
and cannot be expressed in `requirements.txt`, which is why they are code rather
than a setup note.

| # | Failure | Cause | Fix |
| --- | --- | --- | --- |
| 1 | `FileNotFoundError: Could not find module ...robosuite\utils\mujoco.dll` | robosuite opens `mujoco.dll` from its own directory; the wheel ships it inside the `mujoco` package | Copy it across, re-copying when the MuJoCo version changes so a stale DLL is not loaded |
| 2 | `RuntimeError: invalid value for environment variable MUJOCO_GL: egl` | `macros.MUJOCO_GPU_RENDERING = True` forces `egl`, then robosuite's own validator rejects `egl` on Windows. **robosuite 1.4.1 always crashes on Windows at default settings** | Write `robosuite/macros_private.py` with `MUJOCO_GPU_RENDERING = False`, leaving `MUJOCO_GL` alone. Still GPU-rendered |
| 3 | `AttributeError: 'MjData' object has no attribute 'qM'. Did you mean: 'M'?` | mujoco 3.13.0 renamed `MjData.qM` → `M`; robosuite 1.4.1 uses the old name | Pin `mujoco==3.1.6` |
| 4 | `RuntimeError: CMake must be installed` building `egl_probe` | robomimic depends on `egl_probe`, which compiles against EGL headers — unbuildable on Windows and pointless, since there is no EGL. Imported unconditionally by `robomimic/envs/env_robosuite.py:70` whenever offscreen rendering is on, which is the Phase 3 replay path | Install robomimic with `--no-deps` plus its real dependencies, and register a stub `egl_probe` returning an empty device list. Empty is the correct answer here |

`pip check` reports `robomimic 0.3.0 requires egl_probe>=1.0.1, which is not
installed`. Expected, and handled by workaround 4.

---

## Findings carried forward

| Finding | Matters in | Note |
| --- | --- | --- |
| CUDA must be initialised before the render context | every phase | Enforced by `configure_rendering()`. Any script that bypasses it silently loses 6.4× |
| `IMAGE_CONVENTION = "opengl"` is robosuite's default (`robosuite/macros.py:28`) | Phase 2 | Exactly the vertical-flip question `phase-2-zero-shot.md` §2.2 warns about. Left at the default; must be verified against SmolVLA's expectation, not assumed. Switching to `"opencv"` is a one-line macro change |
| Resolution is nearly free; camera count is cheap too | Phase 3 | Choose both on correctness grounds, not cost |
| `lerobot[dataset]` pulls `torchcodec>=0.7` on win32 | Phase 3 | Not installed yet. LeRobot video decoding on Windows is an untested risk — verify at the start of Phase 3 |
| `lerobot` requires `numpy>=2.0,<2.3` | — | Resolved. robosuite 1.4.1 verified working under numpy 2.2.6, so one venv suffices |
| robosuite 1.4.1 pinned over 1.5.2 | Phase 3 | robomimic 0.3.0 pairs with 1.4.x; 1.5.x changed the controller API |
| Observation contract confirmed | Phase 2 | `robot0_eye_in_hand_image`, `agentview_image` as `(res, res, 3) uint8`; `robot0_proprio-state` 32-dim; `action_dim=7` under OSC_POSE |
| Render context costs 0.36 GiB VRAM | Phases 4, 5 | Counts against the 5.0 GiB budget |

---

## Reproducing

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install --no-deps robomimic==0.3.0
python scripts/check_env.py
```

`scripts/check_env.py` is the regression check for the whole repository. Run it at
the start of every later phase.
