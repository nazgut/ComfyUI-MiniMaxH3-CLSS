# ComfyUI-MiniMaxH3-CLSS

**Closed-Loop Streaming Synthesis (CLSS)** for **MiniMax H3** (Hailuo 3.0) — arbitrary-length audio-video generation in [ComfyUI](https://github.com/comfyanonymous/ComfyUI), on consumer **16 GB VRAM** hardware. Port of the [LTX-2.3 CLSS package](https://github.com/nazgut/ComfyUI-LTX2.3-CLSS) to the H3 architecture.

## What is CLSS?

Video diffusion transformers generate only a few seconds per pass (H3's trained range is ~5–15 s). Naive chunking fails within a few hundred frames: the model consumes its own slightly off-distribution output and exposure-bias drift compounds into scene collapse.

CLSS treats the chunk hand-off as a **feedback loop** and controls it. Chunks share a streaming latent buffer (**SLB**) overlap and between chunks CLSS applies lightweight corrections — **without modifying any transformer weights**:

- **Calibrated context re-noising** (τc) — the video overlap is written into the chunk's initial latent and re-noised via H3's per-token denoise masks (mask m → per-row sigma m·σ), so the model actively re-projects it onto the data manifold instead of accepting it verbatim
- **EMA-tracked per-channel AdaIN** (β) — suppresses fast statistical drift; the EMA reference **resets at every scene change**
- **Dynamic anchor bank** — long-range identity tracking; top-m anchors are pinned as H3 `minimax_keyframes` conditioning rows (re-injected every step, never denoised)
- **Two-band spatial detail anchor** — counters progressive high-frequency decay on long runs
- **Audio seam guide** — the last N seconds of the previous chunk's audio are pinned as a `cond_audio` guide keyframe whose window **ends exactly at the join and reaches backward** (fractional/negative anchor index). This end-aligned placement is the measured mechanism that takes seam correlation from 0.45 to 0.95+; a forward/overlap-aligned guide makes the model loop the motif instead. The guide is the *only* audio context — overlap rows are fresh noise and the join is a plain cut
- **Split video/audio CFG with continuation-chunk falloff** — H3 ships one scalar CFG over the packed AV output; the CLSS guider unpacks the stream and applies video_cfg / audio_cfg separately, with rescale. On continuation chunks `audio_cfg_cont` drops audio CFG (default 1.0 = off): the SLB overlap cancels out of the CFG direction, so high audio CFG at a join just amplifies the re-applied text prompt and opens a new musical section every chunk
- **Optional i2v first-frame guide** — an image input is VAE-encoded and pinned as a `minimax_keyframes` row at frame 0 of chunk 0 (H3-native first-frame conditioning)
- **Per-scene R2V references** — H3's ref2va mechanism split by scene: reference images/audios bind to `<Picture N>` / `<Audio N>` labels in one scene's prompt and ride only that scene's chunks

## Multi-scene prompts

`CLSSH3ScenePrompts` takes one prompt per scene, separated by a line containing only `---`. Scenes are unpacked proportionally across `num_chunks`; boundaries use a **two-step crossfade** (`scene_handoff="transition_chunk"`): the outgoing scene's last chunk is guided by a 25%-incoming embedding blend, the incoming scene's first chunk by 75%-incoming. Rule of thumb: `num_chunks ≥ 2 × scenes`.

Note: H3's RoPE t-origin sits after the text span, so a scene's text is reused verbatim across its chunks (position stability); the crossfade blends only at boundaries.

**Prompt format matters.** H3 (and the ClipProj projection) is calibrated on MiniMax's
structured six-section format: `subject_definitions:` / `summary:` /
`detailed_description:` / `[Shot N] timecode-timecode.` / `overall_soundscape:` /
`non_diegetic_music:`. Long free-form prose measurably degrades output. Each CLSS scene
block must carry the full structure (a chunk window only ever sees its own scene's
text); keep each block under ~400 words. The canonical workflow's 3-scene Ferrari
prompt is written in this format — copy its shape.

## R2V references (per-scene image/audio anchors)

`CLSSH3SceneReference` / `CLSSH3SceneReferences` attach reference media to **one scene's** conditioning (chain one node per scene; refs never leak across scenes). The scene text is re-tokenized with the reference presentation, so labels bind at tokenize time — reference them in the prompt as `<Picture 1..N>` for images and `<Audio 1..M>` for audios, in socket order. This is H3's native ref2va mechanism made per-scene: identity/style anchors follow their scene's chunks only.

The multi-ref node uses V3 Autogrow sockets (up to 9 images + 3 audios per scene); the single-ref node stacks one image and/or one audio per node. `ref_image_size=match` (default) downscales refs to the generation's pixel area — `max` keeps more identity detail but ref tokens ride **every** chunk of the scene and can be several times slower. See [`workflow/t2v_with_ref_minimaxh3_clss.json`](workflow/t2v_with_ref_minimaxh3_clss.json) for the full chain.

## Model files

From `Comfy-Org/MiniMax-H3` on Hugging Face:

| File | Place in |
|---|---|
| `minimax_h3_fl2va_int8_convrot.safetensors` | `models/diffusion_models/` |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

Text encoder — two options:

- **Small (recommended for 16 GB cards):** the [ComfyUI-ClipProj](https://github.com/NicoLab28) pack's `ClipProjLoader` with a Qwen3-VL-4B (`qwen3vl_4b_fp8_scaled.safetensors`, `models/text_encoders/`) + learned projection (`mmh3-4b-ClipProj-v3.1.safetensors`, `models/clip_projections/`). ~5.5 GB instead of 15.7 GB; the canonical workflow uses this.
- **Stock:** `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` via `CLIPLoader` (type `minimax`) — swap node 4 in the workflow if you prefer the full 32B encoder.

Requires **ComfyUI ≥ 0.34** (the audio seam guide relies on fractional/negative keyframe anchor indices; the sampler fails loudly with instructions on older versions when the guide is enabled).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nazgut/ComfyUI-MiniMaxH3-CLSS.git
```

Restart ComfyUI — no pip install step, no submodules. Two workflows ship in `workflow/`:

- [`t2v_minimaxh3_clss.json`](workflow/t2v_minimaxh3_clss.json) — canonical t2v, the live-validated 16 GB reference config: 832×480, 243 px ≈ 10 s chunk windows, 10 chunks ≈ 100 s total, 20 steps, sigma shift 12/6.
- [`t2v_with_ref_minimaxh3_clss.json`](workflow/t2v_with_ref_minimaxh3_clss.json) — R2V variant: 3 scenes × 1 chunk at 640×320, each scene anchored by a `CLSSH3SceneReferences` chain (3 identity images per scene).

## Nodes

Every input carries an in-UI tooltip with its default behavior and the evidence behind it.

| Node | Purpose |
|---|---|
| **CLSS H3 Config** | CLSS hyperparameters (τc, β, overlap on the 5k+2 token grid) |
| **CLSS H3 Scene Prompts** | Per-scene prompts (split on `---`) → multi-entry CONDITIONING; stashes raw scene text for the ref nodes |
| **CLSS H3 Scene Reference (R2V)** | Attach one reference image and/or audio to one scene's conditioning (`<Picture N>` / `<Audio N>` labels) |
| **CLSS H3 Scene References (R2V multi)** | All of one scene's refs in one node — V3 Autogrow sockets, up to 9 images + 3 audios, socket order = label order |
| **CLSS H3 Streaming Sampler** | The chunked sampler — SLB via denoise masks, anchor keyframe rows, end-aligned audio seam guide, scene crossfade, optional i2v first-frame guide, corrections, per-chunk telemetry + end-of-run trend summary |
| **CLSS H3 Guider** | Split video/audio CFG + rescale over the packed AV stream |
| **CLSS H3 Video Decode+Save** | Streaming temporal-slice video decode straight to PNG frames on disk + audio decode |

```
UNETLoader → CLSSH3Guider ← CLSSH3ScenePrompts(+) [→ CLSSH3SceneReference(s) per scene] / CLSSH3ScenePrompts(−)
EmptyMiniMaxH3LatentAV → CLSSH3StreamingSampler (+ CLSSH3Config, KSamplerSelect, BasicScheduler, RandomNoise)
→ CLSSH3VideoDecodeSave → PNG frames + AUDIO
```

## Repository layout

```
nodes.py     # all 7 ComfyUI node implementations
clss.py      # the model-agnostic CLSS algorithm core (SLB, EMA/AdaIN, anchor bank)
workflow/    # canonical t2v + R2V workflows — copy them for experiments, don't mutate in place
```

## Status

Live-validated on the 16 GB reference stack (int8 convrot DiT, ClipProj Qwen3-VL-4B text encoder, 832×480, 243 px windows, 20 steps, sigma shift 12/6). Audio seam continuity is measured, not guessed: the end-aligned guide takes cross-join correlation from 0.45 to 0.95+, and per-chunk telemetry (`aud_bnd` / `aud_dlv` / `aud_lvl` / …) localizes any remaining seam or drift issues. Defaults are the measured production config — read the tooltips before changing them.

## Support

If this node pack is useful to you, you can support its development on Patreon: **[patreon.com/c/AleksanderM](https://www.patreon.com/c/AleksanderM)**

## Acknowledgements

Built on [MiniMax H3](https://huggingface.co/Comfy-Org/MiniMax-H3) by MiniMax (weights under the MiniMax H3 Community License — read it before commercial use), [LTX-2](https://github.com/Lightricks/LTX-2) by Lightricks, and the ComfyUI ecosystem.
