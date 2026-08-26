# AGENTS.md

Guidance for AI coding agents working in this repository. Read this before making any change.

## Project overview

**ComfyUI-MiniMaxH3-CLSS** is a **ComfyUI custom-node package** implementing CLSS (Closed-Loop
Streaming Synthesis) — arbitrary-length audio-video generation with the MiniMax H3 (Hailuo 3.0)
omni-modal model on consumer 16 GB VRAM hardware. It is a port of the LTX-2.3 CLSS package
(github.com/nazgut/ComfyUI-LTX2.3-CLSS). Loaded by ComfyUI directly from
`ComfyUI/custom_nodes/<name>`; there is no package manifest and no submodule — the CLSS
algorithm core is vendored as `clss.py`.

CLSS generates video in short temporal **chunks** sharing an **SLB** (streaming latent buffer)
overlap, keeping latent memory O(overlap) instead of O(length). Between chunks it applies
closed-loop corrections that fight exposure-bias drift **without modifying transformer
weights**:

- **§2.1** calibrated context re-noising (`tau_c`, default 0.05; per-chunk schedule rises
  toward a 0.10 ceiling with a 5-chunk half-life) — on H3 implemented via per-token
  denoise masks (mask m → per-row sigma m·σ, `comfy/ldm/minimax/model.py` `_forward`)
- **§2.3** EMA-tracked per-channel AdaIN drift correction (`beta`, default 0.4; the EMA
  reference resets at every scene change via `CLSSState.reset_drift_refs`)
- **§2.5** dynamic anchor bank — on H3 the top-m anchors are pinned as
  `minimax_keyframes` conditioning rows at `resolved_frame_index=0` of the chunk window
  (the LTX version tracked the bank for telemetry only; H3's keyframe-row mechanism is
  what makes real anchor conditioning expressible)
- Two-band spatial detail anchor; audio SLB seam modes (see `audio_slb_tau_mult` below)

## H3 architecture facts that shape the code

- Latent = one dict with `NestedTensor((video [B,24,T,H/16,W/16], audio [B,32,2,Ta]))`.
  Video: 24 ch, 16× spatial, **17k+5 px-frame ↔ 5k+2 latent-token** grid at 24 fps.
  Audio: 32 ch × 2 stereo lanes, **40 latent fps**, time is the LAST axis. All chunk
  math lives on these grids — see the grid helpers and their unit-test-verified comments
  in `nodes.py`.
- Flow matching, `ModelType.FLOW_AV` + `ModelSamplingAV`: the sampler carries audio on
  the video sigma schedule scaled by `audio_scale = shift/audio_shift` (defaults
  shift 12.0 / audio_shift 3.0, overridable on the stock `MiniMaxH3SigmaShift` node).
  There is no token-count-dependent shift (that was LTXVScheduler) — hence no
  `audio_shift_mult` knob here.
- **No hard RoPE wall** (no `max_pos`), but the trained range is ~124–362 px frames
  (~5–15 s). `nodes.py` enforces a soft 12 s window cap (`_WINDOW_CAP_S`) with
  grid-aligned auto-split, same logic shape as the LTX RoPE-wall enforcement.
- RoPE t-axis units are audio latent frames (1/40 s) and the **t-origin sits after the
  text span** — a scene's text must stay byte-identical across its chunks; the scene
  crossfade blends embeddings only at boundary chunks.
- Guide/anchor injection is **conditioning-space** (`minimax_keyframes` rows, pinned
  near-clean, re-injected every step, never denoised) — never write guides into the
  denoised latent stream on H3. The SLB overlap is the one exception that does go into
  the initial latent, paired with the denoise mask (that is the τc lever).
- Stock sampling path is external: `KSamplerSelect + BasicScheduler + guider +
  RandomNoise → SamplerCustomAdvanced` (driven per chunk internally by the sampler node).
- Video VAE tiles internally; audio VAE decode is `VAEDecodeAudio` on the unbound audio
  stream. Batch size 1 only.

## Repository layout

```
nodes.py     # all 5 ComfyUI node implementations
clss.py      # model-agnostic CLSS core: CLSSConfig, CLSSState (SLB, §2.3 EMA/AdaIN,
             # §2.5 anchor bank, post_process, reset_drift_refs) — no ltx imports
__init__.py  # node-mapping exports only
workflow/    # canonical workflow: t2v_minimaxh3_clss.json (API format).
             # RULE: every experiment copies the canonical file — never mutate it in place.
```

### Relationship to the LTX repo

The LTX repo owns the LTX-2.3 implementation and its own `Ltx-2-CLSS` submodule. This
repo's `clss.py` was vendored from that submodule (LTX-specific conditioning methods
removed, `overlap_latent`/`top_anchors` accessors added). If the algorithm core changes
materially, port the change both ways by hand — there is no shared dependency.

## The nodes and how they wire together

```
UNETLoader / CLIPLoader(minimax) / VAELoader×2 (video fp16, audio fp32)
CLSSH3ScenePrompts(CLIP, prompts)          → CONDITIONING (one entry per scene, '---' split;
                                             used twice: positive scenes + negative)
CLSSH3Guider(model, pos, neg, video_cfg 4.0, audio_cfg 7.0, rescale 0.7) → GUIDER
EmptyMiniMaxH3LatentAV (stock)             → LATENT (per-chunk AV template, 5k+2 grid)
CLSSH3Config                               → CLSS_CONFIG
KSamplerSelect + BasicScheduler + RandomNoise → SAMPLER / SIGMAS / NOISE
CLSSH3StreamingSampler(...)                → LATENT  (chunked, full telemetry)
CLSSH3VideoDecodeSave(vae, audio_vae, ...) → PNG frames on disk + AUDIO
```

`scene_handoff`: `transition_chunk` (default) = two-step crossfade straddling each
boundary (outgoing block's last chunk 25%-incoming, incoming block's first chunk
75%-incoming; needs every scene block ≥2 chunks, i.e. `num_chunks ≥ 2×scenes`);
`blend` = single 50/50 chunk; `hard` = plain text swap. EMA/refs reset on the first
incoming-leaning chunk.

Notable sampler knobs (defaults are the starting config, NOT yet live-validated on H3):
`detail_anchor` on; `video_slb_tau_mult` 1.0; `audio_slb_tau_mult` 0.0 (0 = audio SLB
frozen; >0 = re-noised, ceiling 0.35; <0 = overlap free except the last |value| SECONDS
of the previous tail pinned at mask 0 at the end of the overlap — keeps vocal phrases
glued across the seam); `fps` forced to 24 with a warning (the 40-latent-fps audio math
is hard-wired to 24).

## Build, run, and test commands

- **Run in ComfyUI** (the ground truth): `cd ../.. && python main.py`, load
  `workflow/t2v_minimaxh3_clss.json`. The generation path can only be validated live.
- **Import smoke test** (no GPU work):
  `cd /home/n/AI/ComfyUI && myenv/bin/python -c "import sys; sys.path.insert(0,'custom_nodes/ComfyUI-MiniMaxH3-CLSS'); import nodes; print(sorted(nodes.NODE_CLASS_MAPPINGS))"`
- `myenv/bin/python -m py_compile nodes.py clss.py` before committing. There is no
  test suite; the subagent that wrote the port unit-checked the grid math, the
  crossfade blend and the noise slicing — keep those invariants (17k+5 / 5k+2 grid
  alignment, cumulative-absolute audio positions, exact N(0,1) marginals for
  `noise_temporal_corr`) covered by comments at minimum.

## Conventions specific to this codebase

- **Node inputs are experiment knobs, not user settings.** Defaults are the intended
  production config. **Read the tooltip/docstring before changing a default.**
- **Removing a failed experiment means deleting its input + code**, not defaulting it off.
- **Latent metrics measure structure only.** They localize failures; they never prove a
  quality win. The user's eyes/ears on a live decode are the only ground truth.
- **The denoising/generation path is high-risk.** Never ship a change to the chunk loop,
  mask construction, or correction math without a user-validated live run. Noise edits
  are only seed-safe if they preserve the exact N(0,1) marginal.
- Tooltips on every input; heavy docstrings citing §-sections and measured evidence;
  every non-obvious constant carries its justification.

## Security considerations

- This package executes inside ComfyUI's Python process with full user privileges and
  loads multi-GB model weights from local paths. Never add network fetches or dynamic
  code loading at import time.
- Do not commit model files, generated videos, or secrets. `.gitignore` covers Python
  artifacts, venvs, `*.log`, `tmp/`.
- H3 weights are under the MiniMax H3 Community License (territorial/commercial
  restrictions) — mention it when redistributing.
