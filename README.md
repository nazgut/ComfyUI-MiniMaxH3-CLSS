# ComfyUI-MiniMaxH3-CLSS

**Closed-Loop Streaming Synthesis (CLSS)** for **MiniMax H3** (Hailuo 3.0) — arbitrary-length audio-video generation in [ComfyUI](https://github.com/comfyanonymous/ComfyUI), on consumer **16 GB VRAM** hardware. Port of the [LTX-2.3 CLSS package](https://github.com/nazgut/ComfyUI-LTX2.3-CLSS) to the H3 architecture.

## What is CLSS?

Video diffusion transformers generate only a few seconds per pass (H3's trained range is ~5–15 s). Naive chunking fails within a few hundred frames: the model consumes its own slightly off-distribution output and exposure-bias drift compounds into scene collapse.

CLSS treats the chunk hand-off as a **feedback loop** and controls it. Chunks share a streaming latent buffer (**SLB**) overlap and between chunks CLSS applies lightweight corrections — **without modifying any transformer weights**:

- **Calibrated context re-noising** (τc) — the overlap is written into the chunk's initial latent and re-noised via H3's per-token denoise masks (mask m → per-row sigma m·σ), so the model actively re-projects it onto the data manifold instead of accepting it verbatim
- **EMA-tracked per-channel AdaIN** (β) — suppresses fast statistical drift; the EMA reference **resets at every scene change**
- **Dynamic anchor bank** — long-range identity tracking; top-m anchors are pinned as H3 `minimax_keyframes` conditioning rows (re-injected every step, never denoised)
- **Two-band spatial detail anchor** — counters progressive high-frequency decay on long runs
- **Audio seam control** — the audio SLB can be placed frozen, re-noised, or regenerated freely with the last N seconds of the previous tail pinned at the seam so vocal phrases aren't cut mid-phoneme
- **Optional temporally-correlated noise** (FreeNoise/PYoCo family, exact N(0,1) marginals preserved)
- **Split video/audio CFG** — H3 ships one scalar CFG over the packed AV output; the CLSS guider unpacks the stream and applies video_cfg / audio_cfg separately, with rescale

## Multi-scene prompts

`CLSSH3ScenePrompts` takes one prompt per scene, separated by a line containing only `---`. Scenes are unpacked proportionally across `num_chunks`; boundaries use a **two-step crossfade** (`scene_handoff="transition_chunk"`): the outgoing scene's last chunk is guided by a 25%-incoming embedding blend, the incoming scene's first chunk by 75%-incoming. Rule of thumb: `num_chunks ≥ 2 × scenes`.

Note: H3's RoPE t-origin sits after the text span, so a scene's text is reused verbatim across its chunks (position stability); the crossfade blends only at boundaries.

## Model files

From `Comfy-Org/MiniMax-H3` on Hugging Face:

| File | Place in |
|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

Requires ComfyUI ≥ 0.30 with native MiniMax H3 support (nodes `EmptyMiniMaxH3LatentAV` etc.).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nazgut/ComfyUI-MiniMaxH3-CLSS.git
```

Restart ComfyUI — no pip install step, no submodules. Load [`workflow/t2v_minimaxh3_clss.json`](workflow/t2v_minimaxh3_clss.json) (canonical t2v config: 1344×768, 11.5 s chunk windows, 6 chunks ≈ 65 s).

## Nodes

Every input carries an in-UI tooltip with its default behavior and the evidence behind it.

| Node | Purpose |
|---|---|
| **CLSS H3 Config** | CLSS hyperparameters (τc, β, overlap on the 5k+2 token grid, noise temporal correlation) |
| **CLSS H3 Scene Prompts** | Per-scene prompts (split on `---`) → multi-entry CONDITIONING |
| **CLSS H3 Streaming Sampler** | The chunked sampler — SLB via denoise masks, anchor keyframe rows, scene crossfade, corrections, per-chunk telemetry + end-of-run trend summary |
| **CLSS H3 Guider** | Split video/audio CFG + rescale over the packed AV stream |
| **CLSS H3 Video Decode+Save** | Streaming temporal-slice video decode straight to PNG frames on disk + audio decode |

```
UNETLoader → CLSSH3Guider ← CLSSH3ScenePrompts(+) / CLSSH3ScenePrompts(−)
EmptyMiniMaxH3LatentAV → CLSSH3StreamingSampler (+ CLSSH3Config, KSamplerSelect, BasicScheduler, RandomNoise)
→ CLSSH3VideoDecodeSave → PNG frames + AUDIO
```

## Repository layout

```
nodes.py     # all 5 ComfyUI node implementations
clss.py      # the model-agnostic CLSS algorithm core (SLB, EMA/AdaIN, anchor bank)
workflow/    # canonical t2v workflow — copy it for experiments, don't mutate in place
```

## Status

Early port: the generation path is validated by import/logic checks only until live runs confirm it. Defaults follow the LTX-validated CLSS config where the mechanism maps 1:1; H3-specific choices (12 s window cap, overlap 7 tokens, keyframe-row anchors) are conservative starting points, not yet ear/eye-validated.

## Acknowledgements

Built on [MiniMax H3](https://huggingface.co/Comfy-Org/MiniMax-H3) by MiniMax (weights under the MiniMax H3 Community License — read it before commercial use), [LTX-2](https://github.com/Lightricks/LTX-2) by Lightricks, and the ComfyUI ecosystem.
