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
- **Optional audio recompose** — per chunk, the generated audio is discarded and re-imagined from **pure noise** against the finished chunk video (downscaled frozen video reference + masked dummy target, so a step costs seconds instead of a full chunk), optionally by a separate **BASE-model** guider (`audio_refine_guider`). Fresh noise is what changes the take — re-noising measured cos 0.90–0.96, i.e. the same take — and turbo LoRAs are video-distilled, so recomposing with the turbo head re-cooks the same under-distilled audio. See [`workflow/t2v_lora_minimaxh3_clss.json`](workflow/t2v_lora_minimaxh3_clss.json)
- **Split video/audio CFG** — H3 ships one scalar CFG over the packed AV output; the CLSS guider unpacks the stream and applies video_cfg / audio_cfg separately, with rescale, uniformly to every chunk. (The SLB overlap cancels out of the CFG direction, so high audio CFG at a join mainly amplifies the re-applied text prompt — measured to open a new musical section every chunk; keep `audio_cfg` at 1.0 unless experimenting.)
- **Optional i2v first-frame guide** — an image input is VAE-encoded and pinned as a `minimax_keyframes` row at frame 0 of chunk 0 (H3-native first-frame conditioning)
- **Per-scene R2V references** — H3's ref2va mechanism split by scene: reference images/audios bind to `<Picture N>` / `<Audio N>` labels in one scene's prompt and ride only that scene's chunks; the all-scenes node fans images out to every scene and cuts a soundtrack into consecutive per-scene windows

## Multi-scene prompts

`CLSSH3ScenePrompts` takes one prompt per scene, separated by a line containing only `---`. Scenes are unpacked proportionally across `num_chunks`; boundaries use a **two-step crossfade** (`scene_handoff="transition_chunk"`): the outgoing scene's last chunk is guided by a 25%-incoming embedding blend, the incoming scene's first chunk by 75%-incoming. Rule of thumb: `num_chunks ≥ 2 × scenes`.

**Shared text once.** The node's optional `global_text` field is copied to the **top of every scene block** before encoding — put style, `subject_definitions`, `overall_soundscape`, `non_diegetic_music` or quality rules there instead of repeating them in each `---` block. The prefix is baked into each scene's text, so it stays byte-identical across that scene's chunks (RoPE position stability) and survives the R2V nodes' re-tokenization. Leave it empty for the old behavior.

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

**All scenes at once.** `CLSSH3SceneReferencesAll` removes the per-scene chain: every connected `ref_image` is attached to **all** scenes, and the connected `ref_audio` file(s) are concatenated and cut into consecutive windows, one per scene — scene *i* anchors on seconds `[i·T, (i+1)·T)` of the track ("10 s after 10 s"), `T = audio_seconds_per_scene` (default 10 s). Set `T` to the time one scene actually generates (chunks per scene × chunk length) so each scene is anchored to the musical/voice span it is producing. Scenes past the end of the track keep their image refs only; a warning prints which scenes lost their audio window. Change the `---` scene list and nothing needs rewiring — the canonical R2V workflow uses this node.

## Upscaling (neural latent upscaler)

[Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) plugs into the sampler **chunk by chunk** — that pack must be installed (CLSS **soft-imports** its 3D model module at execute time — nothing is vendored) and its checkpoint placed in `models/latent_upscale_models/`.

`CLSSH3LoadLatentUpscaleModel` → `CLSSH3StreamingSampler.upscaler` + `upscale_scale`:

- Every chunk is upscaled **right after its SLB step**, so the streaming state stays low-res and a long video never exists at high resolution all at once.
- Each chunk's **full window** (SLB overlap + new tokens) is sent to the upscaler for left temporal context; the overlapping span is **cross-faded** over the previous chunk's delivered tail, so upscale seams blend instead of stepping.
- The node returns high-res **video** + unchanged **audio** (the upscaler is spatial only; time is preserved). Set `EmptyMiniMaxH3LatentAV` to the **low-res** generation size — output = template × `upscale_scale`, aligned to the 32-px canvas rule (832×480 × 1.5 → 1248×720).
- Costs ~0.7 GB extra VRAM for the upscaler (fp16) during the run; it is moved to the **compute device (CUDA)** at run start and offloaded afterwards. (Without CUDA it falls back to fp32 CPU inference, which is far too slow for this network — don't use it there.) Decoding the bigger latent needs more VRAM per slice — lower `frames_per_slice` in the decode node if needed.

Any slice of the 1.0→0.0 schedule is accepted: a low-res pass may end above zero — its x0 is then what the upscaler carries — and a partial schedule may also start below 1.0, in which case every chunk starts from noise at σ0.

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

Restart ComfyUI — no pip install step, no submodules. Three workflows ship in `workflow/`:

- [`t2v_minimaxh3_clss.json`](workflow/t2v_minimaxh3_clss.json) — canonical t2v, the live-validated 16 GB reference config: 832×480, 243 px ≈ 10 s chunk windows, 10 chunks ≈ 100 s total, 20 steps, sigma shift 12/6.
- [`t2v_with_ref_minimaxh3_clss.json`](workflow/t2v_with_ref_minimaxh3_clss.json) — R2V variant: 3 scenes × 1 chunk at 640×320, all scenes anchored by one `CLSSH3SceneReferencesAll` node (3 identity images on every scene; drop in a soundtrack and it is auto-sliced 10 s per scene).
- [`t2v_lora_minimaxh3_clss.json`](workflow/t2v_lora_minimaxh3_clss.json) — turbo-LoRA variant: `MiniMaxH3TurboLoRA` on the main guider, with a BASE-model `audio_refine_guider` and `audio_recompose_steps=20` recomposing the final audio against the finished video.

## Nodes

Every input carries an in-UI tooltip with its default behavior and the evidence behind it.

| Node | Purpose |
|---|---|
| **CLSS H3 Config** | CLSS hyperparameters (τc, β, overlap on the 5k+2 token grid) |
| **CLSS H3 Scene Prompts** | Per-scene prompts (split on `---`) → multi-entry CONDITIONING; optional `global_text` prepended to every scene; stashes raw scene text for the ref nodes |
| **CLSS H3 Scene Reference (R2V)** | Attach one reference image and/or audio to one scene's conditioning (`<Picture N>` / `<Audio N>` labels) |
| **CLSS H3 Scene References (R2V multi)** | All of one scene's refs in one node — V3 Autogrow sockets, up to 9 images + 3 audios, socket order = label order |
| **CLSS H3 Scene References (R2V all scenes)** | One node for the whole scene list: every image attaches to all scenes, the ref audio is concatenated and cut into consecutive per-scene windows (10 s after 10 s, `audio_seconds_per_scene`); replaces the per-scene chain |
| **CLSS H3 Load Latent Upscale Model** | Loads a Minimax H3 latent-upscaler (3D) checkpoint from `models/latent_upscale_models` for the sampler's per-chunk upscale; the model code is soft-imported from the Comfyui_Minimax_h3_latent_Upscaler pack at execute time |
| **CLSS H3 Streaming Sampler** | The chunked sampler — SLB via denoise masks, anchor keyframe rows, end-aligned audio seam guide, scene crossfade, optional i2v first-frame guide, optional audio recompose against the finished video, optional per-chunk neural upscale (`upscaler` + `upscale_scale`), corrections, per-chunk telemetry + end-of-run trend summary |
| **CLSS H3 Guider** | Split video/audio CFG + rescale over the packed AV stream |
| **CLSS H3 Video Decode+Save** | Streaming temporal-slice video decode straight to PNG frames on disk + audio decode |

```
UNETLoader → CLSSH3Guider ← CLSSH3ScenePrompts(+) [→ CLSSH3SceneReference(s) per scene] / CLSSH3ScenePrompts(−)
EmptyMiniMaxH3LatentAV → CLSSH3StreamingSampler (+ CLSSH3Config, KSamplerSelect, BasicScheduler, RandomNoise)
→ CLSSH3VideoDecodeSave → PNG frames + AUDIO
```

## Repository layout

```
nodes.py     # all 9 ComfyUI node implementations
clss.py      # the model-agnostic CLSS algorithm core (SLB, EMA/AdaIN, anchor bank)
workflow/    # canonical t2v / R2V / turbo-LoRA workflows — copy them for experiments, don't mutate in place
```

## Status

Live-validated on the 16 GB reference stack (int8 convrot DiT, ClipProj Qwen3-VL-4B text encoder, 832×480, 243 px windows, 20 steps, sigma shift 12/6). Audio seam continuity is measured, not guessed: the end-aligned guide takes cross-join correlation from 0.45 to 0.95+, and per-chunk telemetry (`aud_bnd` / `aud_dlv` / `aud_lvl` / …) localizes any remaining seam or drift issues. Defaults are the measured production config — read the tooltips before changing them.

## Updates

**2026-09-12 — sampler option cleanup**

- **Removed sampler options** — `refine_latent` (the two-pass refine), `audio_cfg_cont` (the guider's `audio_cfg` now applies to every chunk), `audio_refresh_waveform` and the sampler's `audio_vae` input (the audio continuity reference is always the delivered latent — the VAE round-trip was lossy and compounded). Dead keys dropped from the canonical workflows.

**2026-09-11 — all-scene refs, global prompt text, audio recompose + seam controls**

- **Chunk-by-chunk upscaling inside the sampler** — `CLSSH3LoadLatentUpscaleModel` + the sampler's `upscaler` / `upscale_scale`: every chunk is upscaled **after its SLB step** (full window in, overlap cross-faded over the previous tail), so a long video never exists at high res at once — the per-chunk, memory-bounded version of [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler). See [Upscaling](#upscaling-neural-latent-upscaler). The sampler also accepts **any slice** of the 1.0→0.0 flow schedule (a low-res pass may end above zero; a partial schedule may also start below 1.0).
- **All-scene R2V refs** — new `CLSSH3SceneReferencesAll` node: every `ref_image` attaches to ALL scenes, and the connected `ref_audio` file(s) are concatenated and cut into consecutive per-scene windows (`audio_seconds_per_scene`, default 10 s → "10 s after 10 s"). Replaces chaining one ref node per `---` block; [`t2v_with_ref_minimaxh3_clss.json`](workflow/t2v_with_ref_minimaxh3_clss.json) now uses it.
- **`global_text` on `CLSSH3ScenePrompts`** — one text field copied to the top of every scene block before encoding, so shared style/section text is written once; the prefix is baked into each scene's text and survives the ref nodes' re-tokenization.
- **Audio recompose** — `audio_recompose_steps` / `_sigma` / `_pool` / `_stride` / `_seed` + `audio_refine_guider`: per chunk, a fresh audio take from pure noise against the finished video (the measured fix for turbo-LoRA audio). New [`t2v_lora_minimaxh3_clss.json`](workflow/t2v_lora_minimaxh3_clss.json) ships the turbo-LoRA + base-model-recompose config.
- **Audio seam controls** — the `ref_audio` continuity block is now placed through the H3 layout patch so it **ends exactly at the join** (length and end position decoupled); `audio_head_discard_ms` drops the unstable opening of continuation chunks; `audio_xfade_ms` crossfades the join.
- **Determinism** — cross-chunk noise fields are generated at fixed caps, so chunk 1 is bit-identical regardless of `num_chunks` (`torch.randn` has no prefix property).
- **Cleanup** — dead guide-layout check removed; the long inline design notes in `nodes.py` trimmed down to the measured facts (details live in this README and `AGENTS.md`).

## Support

If this node pack is useful to you, you can support its development on Patreon: **[patreon.com/c/AleksanderM](https://www.patreon.com/c/AleksanderM)**

## Acknowledgements

Built on [MiniMax H3](https://huggingface.co/Comfy-Org/MiniMax-H3) by MiniMax (weights under the MiniMax H3 Community License — read it before commercial use), [LTX-2](https://github.com/Lightricks/LTX-2) by Lightricks, and the ComfyUI ecosystem.
