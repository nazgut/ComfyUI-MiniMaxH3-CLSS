# ComfyUI-MiniMaxH3-CLSS

**Closed-Loop Streaming Synthesis (CLSS)** for **MiniMax H3** (Hailuo 3.0) — arbitrary-length audio-video generation in [ComfyUI](https://github.com/comfyanonymous/ComfyUI), on consumer **16 GB VRAM** hardware. Port of the [LTX-2.3 CLSS package](https://github.com/nazgut/ComfyUI-LTX2.3-CLSS) to the H3 architecture.

## What is CLSS?

Video diffusion transformers generate only a few seconds per pass (H3's trained range is ~5–15 s). Naive chunking fails within a few hundred frames: the model consumes its own slightly off-distribution output and exposure-bias drift compounds into scene collapse.

CLSS treats the chunk hand-off as a **feedback loop** and controls it. Chunks share a streaming latent buffer (**SLB**) overlap and between chunks CLSS applies lightweight corrections — **without modifying any transformer weights**:

- **Calibrated context re-noising** (τc) — the video overlap is written into the chunk's initial latent and re-noised via H3's per-token denoise masks (mask m → per-row sigma m·σ), so the model actively re-projects it onto the data manifold instead of accepting it verbatim
- **EMA-tracked per-channel AdaIN** (β) — suppresses fast statistical drift; the EMA reference **resets at every scene change**
- **Dynamic anchor bank** — long-range identity tracking; top-m anchors are pinned as H3 `minimax_keyframes` conditioning rows (re-injected every step, never denoised)
- **Two-band spatial detail anchor** — counters progressive high-frequency decay on long runs
- **Audio continuity context** — the previous chunk's audio tail rides as a
`minimax_refs` reference block whose placement **ends exactly at the join**.
With the sampler's optional `audio_vae` input wired (all bundled workflows),
the ref is **refreshed at every boundary from the audible result** — the
delivered tail is decoded, normalized exactly like the export chain, cut to
the span and re-encoded (Motion Director's `audio_context_refresh` normal
path: the model continues from what the ears will hear, not from the
generated latent's hidden state, which recursively drifts). Without the VAE
wiring — or when the waveform cannot cover the span — the DELIVERED latent is
carried as the strict fallback (MD `_waveform_can_refresh` order; the chunk
log marks the source `(wav)`/`(lat)`; `audio_ref_wav_refresh=false` skips the
round-trip entirely, and the re-encoded ref is re-anchored to the RMS of the
delivered tail window it was decoded from so the refresh cannot leak ~2 dB of
loudness per chunk). The block LENGTH follows the video
context span (the overlap) for the joint ref; `audio_recompose_ref_ms`
lengthens the recompose pass's ref. The seam is a sample-exact trim + an
**equal-power** take-swap crossfade over the span both takes re-rendered
(a content-exact click fix — same wall-clock render on both sides, sin/cos so
a phase-mismatched blend cannot dip -3 dB — never a gain), Motion Director
parity otherwise (MD's audio seam has no gain automation:
`audio_trim.py` / `audio_export.py`, and `segment_continuity.py` is
video-only). Level matching is **export-only**: the recompose take's
delivered-region waveform RMS is anchored to the previous chunk's delivered
waveform (±3 dB clamp; the first chunk sets the song level) and the per-chunk
gain is applied in the decode/save splice — never written back into the
latent or the ref chain (that measured the delivered latent RMS climbing
0.48 → 0.85 over five chunks). The old crossfade, junction lead,
handoff fade, join glide, attack tame and the junction-anchored level
matcher (with its 6 dB content gate) are all removed — the matcher on
2026-09-17 after it measured ducking the reference track's own section
change by 16.7 dB
- **Optional audio recompose** — per chunk, the generated audio is discarded and re-imagined from **pure noise** against the finished chunk video (downscaled frozen video reference + masked dummy target, so a step costs seconds instead of a full chunk), optionally by a separate **BASE-model** guider (`audio_refine_guider`). Fresh noise is what changes the take — re-noising measured cos 0.90–0.96, i.e. the same take — and turbo LoRAs are video-distilled, so recomposing with the turbo head re-cooks the same under-distilled audio. Off in the canonical workflows (`CLSSH3AudioConfig.audio_recompose_steps=0`); `sim/make_audio_lab.py` generates a lab driver that exercises it. Each take is measured for self-repetition; takes that read as a vamp are re-rolled (optional ref-span rescue) and the least-repetitive take is kept. The pass can run on the base weights without a second checkpoint via `CLSSH3BaseRefineGuider`, gains a lot from its own solver (`audio_recompose_sampler`, unwired = inherit the main pick; `CLSSH3AudioConfig.audio_recompose_scheduler`, default `beta`), and its window's overlap span is **pinned to the delivered tail latent** (`audio_seam_pin`) so the take continues through the seam; its video ref is anti-aliased in space and time (stride groups averaged, kept tokens re-spaced to their group's true time centre).
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
text); keep each block under ~400 words. The canonical t2v workflow's Ferrari
prompt is written in this format — copy its shape.

## R2V references (per-scene image/audio anchors)

`CLSSH3SceneReference` / `CLSSH3SceneReferences` attach reference media to **one scene's** conditioning (chain one node per scene; refs never leak across scenes). The scene text is re-tokenized with the reference presentation, so labels bind at tokenize time — reference them in the prompt as `<Picture 1..N>` for images and `<Audio 1..M>` for audios, in socket order. This is H3's native ref2va mechanism made per-scene: identity/style anchors follow their scene's chunks only.

The multi-ref node uses V3 Autogrow sockets (up to 9 images + 3 audios + 3 videos per scene); the single-ref node stacks one image and/or one audio per node. **Video references** ride the same mechanism: `ref_videos` accepts VIDEO or IMAGE sockets (frames resampled to 24 fps when the source differs), is referenced in the prompt as `<Video k>`, and the video's own soundtrack auto-attaches as its audio block unless an index-paired `ref_video_audio` overrides it (without `audio_vae` the soundtrack is skipped with a warning — the video still rides). `ref_image_size=match` (default) downscales refs to the generation's pixel area — `max` keeps more identity detail but ref tokens ride **every** chunk of the scene and can be several times slower. See [`workflow/ref2v_minimaxh3_clss.json`](workflow/ref2v_minimaxh3_clss.json) for the full chain.

**All scenes at once.** `CLSSH3SceneReferencesAll` removes the per-scene chain: every connected `ref_image` is attached to **all** scenes, and the connected `ref_audio` file(s) are concatenated and encoded into **guarded windows**, one per scene. At run start the sampler **crops each window to the exact span its scene delivers** (it knows the chunk plan and scene allocation — nothing to enter by hand, nothing to wire on the sampler side), so the reference can never drift against the delivered timeline; only a span outside the ±4 s guard falls back to a re-encode (that one needs the sampler's `audio_vae`; the log names the scene otherwise). The guard is what absorbs the old exact-T lattice bug — a 243-px chunk delivers 10.125 s, continuations 9.91667 s, so an exact-T window slid ~0.84 % (83.3 ms) per chunk against the piece (`sim/sim_ref_window_drift.py`). Scenes past the end of the track keep their image refs only; a warning prints which scenes lost their audio window. Change the `---` scene list and nothing needs rewiring — the canonical R2V workflow uses this node.

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

Restart ComfyUI — no pip install step, no submodules. Five workflows ship in `workflow/` — the t2v / i2v / R2V set on the live-validated 16 GB reference config (832×480, 243 px ≈ 10 s chunk windows, 20 steps, sigma shift 12/6, audio CFG 4), plus the continue / re-edit graphs built from the turbo R2V production stack:

- [`t2v_minimaxh3_clss.json`](workflow/t2v_minimaxh3_clss.json) — text-to-video: the canonical single-scene Ferrari driving shot (10 chunks ≈ 99 s).
- [`i2v_minimaxh3_clss.json`](workflow/i2v_minimaxh3_clss.json) — image-to-video: `LoadImage` wired to the sampler's `image` + `vae` inputs pins the image as an H3 first-frame keyframe on chunk 0 (swap in your image and edit the prompt to match).
- [`ref2v_minimaxh3_clss.json`](workflow/ref2v_minimaxh3_clss.json) — reference-to-video: one `CLSSH3SceneReferencesAll` node attaches the reference image(s) to every scene (`<Picture N>`); drop in a `ref_audio` track and it is auto-sliced 10 s per scene.
- [`continue_minimaxh3_clss.json`](workflow/continue_minimaxh3_clss.json) — keep going: point `filename_prefix` / `audio_prefix` at a finished run's decode-save / save-audio prefixes and generate the next N chunks from its real tail (same prompt text, same template, same overlap). The output is the new span only — append it after the saved frames/audio.
- [`reedit_minimaxh3_clss.json`](workflow/reedit_minimaxh3_clss.json) — fix one chunk without re-running the piece: pick `chunk_index`, `pin_frames` (first+last by default) anchors the saved seam frames, and optionally present the saved window as a video reference. The node reports which saved frames (`start_frame`, `frames`) to replace with the re-render.

## Nodes

Every input carries an in-UI tooltip with its default behavior and the evidence behind it.

| Node | Purpose |
|---|---|
| **CLSS H3 Config** | CLSS hyperparameters (τc, β, overlap on the 5k+2 token grid); optional experimental knobs, both default 0 (off): `overlap_evict_after` (two-phase overlap eviction) and `step_cache_thresh` (skip a DiT forward when the latent has barely moved — the prediction is Taylor-extrapolated along the schedule from the last two computed steps, with a per-tile veto so a calm background can't drag a moving subject through the skip) |
| **CLSS H3 Audio Config** | All audio settings in one node — recompose steps/scheduler/σ/arc margin/pool/stride/seed/ref span, head discard, the loop guard (re-rolls, thresholds, rescue ref span, draft screening), the recompose **seam pin**, the **ref waveform refresh** toggle and the delivery mode (`corrected` vs `raw` bisect); wire into the sampler's `audio_config` (the sampler's own audio widgets were removed) |
| **CLSS H3 Scene Prompts** | Per-scene prompts (split on `---`) → multi-entry CONDITIONING; optional `global_text` prepended to every scene; optional `audio_continuity_text` swapped in only on chunks that carry the continuation ref; stashes raw scene text for the ref nodes |
| **CLSS H3 Scene Reference (R2V)** | Attach one reference image and/or audio to one scene's conditioning (`<Picture N>` / `<Audio N>` labels) |
| **CLSS H3 Scene References (R2V multi)** | All of one scene's refs in one node — V3 Autogrow sockets, up to 9 images + 3 audios + 3 videos, socket order = label order; video refs (`<Video k>`) take VIDEO/IMAGE sockets and auto-attach their own soundtrack unless a paired `ref_video_audio` overrides it |
| **CLSS H3 Scene References (R2V all scenes)** | One node for the whole scene list: every image attaches to all scenes; the ref audio is encoded into **guarded** per-scene windows and **cropped automatically** by the sampler to each scene's exact delivered span (no geometry to enter, nothing to wire on the sampler side; only a span outside the ±4 s guard falls back to a re-encode); replaces the per-scene chain; video refs attach to every scene the same way |
| **CLSS H3 Continue From Video** | Continue a finished run: reads its saved frames + audio by prefix, turns the last overlap span into the opening context (keyframe replay + SLB seed + audio tail ref) for the sampler's `continuation_context` input; the output is the new span only — the 2-token head the token grid needs is folded into the decoded stream and dropped at save (first 5 frames + 8.67 af of audio), so nothing needs trimming by hand |
| **CLSS H3 Re-edit Chunk** | Re-render one chunk of a finished run at its exact original span, head included (context from the frames before it + first/last frame pins copied from the saved video, optional strided video ref of the saved window); outputs `start_frame` / `frames` so the re-render can replace that span in place |
| **CLSS H3 Load Latent Upscale Model** | Loads a Minimax H3 latent-upscaler (3D) checkpoint from `models/latent_upscale_models` for the sampler's per-chunk upscale; the model code is soft-imported from the Comfyui_Minimax_h3_latent_Upscaler pack at execute time |
| **CLSS H3 Attention Override (Sage/Flash)** | Run the DiT attention on SageAttention / FlashAttention 2 / xformers — or the stock pytorch / sub-quad backends for A/B — through ComfyUI's `optimized_attention_override` hook; the backend is smoke-tested once and falls back to the stock path with a warning when a package is missing or a call fails mid-run |
| **CLSS H3 Spectrum Forecast (skip DiT steps)** | Training-free DiT step skipping for turbo runs, adapted from [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) (Han et al., [Spectrum paper](https://arxiv.org/abs/2603.01623)): actual steps capture the packed target-stream hidden state right after the last DiT block; forecast steps bypass every DiT block and inject a Chebyshev/linear extrapolation of the anchors before the native FinalLayer — the per-step adaln modulation stays exact. Wire it between the model loader and the CLSS guider; defaults turn a 6-step turbo chunk into A A F A F A (≈50% of DiT evals skipped per chunk). Fail-closed — another plugin's block replacements or a bad fit fall back to real steps |
| **CLSS H3 Streaming Sampler** | The chunked sampler — SLB via denoise masks, anchor keyframe rows, end-aligned audio seam guide, scene crossfade, optional i2v first-frame guide, optional audio recompose against the finished video (optional dedicated `audio_recompose_sampler`), optional per-chunk neural upscale (`upscaler` + `upscale_scale`), corrections, per-chunk telemetry + end-of-run trend summary |
| **CLSS H3 Guider** | Split video/audio CFG + rescale over the packed AV stream |
| **CLSS H3 Base Refine Guider (weight-swap)** | Guider for the audio recompose pass that runs on BASE weights without a second checkpoint: clones the turbo model and strips its LoRA patches in a shared-weight copy (ComfyUI re-patches in place on the pass swap); the sampler skips its between-chunks unload |
| **CLSS H3 Video Decode+Save** | Streaming temporal-slice video decode straight to PNG frames on disk + audio decode |

```
UNETLoader (+ optional CLSSH3AttentionOverride / CLSSH3SpectrumForecast) → CLSSH3Guider ← CLSSH3ScenePrompts(+) [→ CLSSH3SceneReference(s) per scene] / CLSSH3ScenePrompts(−)
EmptyMiniMaxH3LatentAV → CLSSH3StreamingSampler (+ CLSSH3Config, KSamplerSelect, BasicScheduler, RandomNoise)
→ CLSSH3VideoDecodeSave → PNG frames + AUDIO
```

## Repository layout

```
nodes.py     # all 15 ComfyUI node implementations
clss.py      # the model-agnostic CLSS algorithm core (SLB, EMA/AdaIN drift correction)
workflow/    # canonical t2v / i2v / R2V / continue / re-edit workflows — copy them for experiments, don't mutate in place
```

## Status

Live-validated on the 16 GB reference stack (int8 convrot DiT, ClipProj Qwen3-VL-4B text encoder, 832×480, 243 px windows, 20 steps, sigma shift 12/6). Audio seam continuity is measured, not guessed: the end-aligned guide takes cross-join correlation from 0.45 to 0.95+, and per-chunk telemetry (`aud_bnd` / `aud_dlv` / `aud_lvl` / …) localizes any remaining seam or drift issues. Defaults are the measured production config — read the tooltips before changing them.

## Updates

**2026-10-01 — seam pin, export-only waveform level anchor, anti-aliased video refs, recompose sampler/scheduler**

- **Seam pin (`CLSSH3AudioConfig.audio_seam_pin`, default on)** — the recompose window's overlap span is frozen to the **actually-delivered tail latent** (`noise_mask 0`), so the take is generated as a direct continuation of the delivered audio instead of re-imagining the overlap and being hard-spliced at the join: the join's phase/timbre/groove discontinuity is removed at the source, with the model generating through the seam. The per-chunk log verifies the pin (`seam pin check max|take-pin|=…`, warning when the mask did not apply); delivery still cuts at the sample-exact boundary, so lengths and A/V sync are untouched. Off = legacy splice behaviour.
- **Waveform loudness anchor (export-only)** — latent RMS stays flat while the decoded waveform gets quieter chunk over chunk (measured ~-2 dB per chunk). The recompose take is now decoded, its *delivered* region RMS'd (past the overlap cut, before the tail margin) and matched to the previous chunk's delivered waveform (±3 dB clamp; the first chunk sets the song level). The gain is recorded per chunk and applied **only in the decode/save splice** — never written back into the latent or the ref chain (applying it there measured the delivered latent RMS climbing 0.48 → 0.85 over five chunks). `audio_delivery="raw"` disables the whole delivery-correction stack (peak soft-clip, scene-ref and spectral gains, the anchor) so a raw take can be A/B-ed against the corrected chain.
- **Waveform-ref unity level** — the `(wav)` ref refresh now re-anchors the re-encoded ref to the RMS of the delivered tail window it was decoded from (log label `wav(lvl x1.23)`), killing the ~2 dB-per-chunk loudness loss the round-trip compounds; `audio_ref_wav_refresh=false` skips the round-trip entirely (the as-delivered corrected latent rides as the ref).
- **Equal-power seam crossfade** — the take-swap blend over the span both takes re-rendered is now sin/cos instead of a linear gain ramp: phase-mismatched takes can dip up to -3 dB in the middle of a linear blend; equal-power keeps seam loudness constant (same content-exact fix, still length-neutral).
- **Anti-aliased temporal decimation** — `audio_recompose_stride` (and the re-edit video ref) no longer decimate with plain `::stride`, which dropped fast transients phase-dependently and made kept tokens read as multiplied apparent motion (the model times hits/lip transients off an aliased signal). Stride groups are now **averaged**, and the layout fixup places each kept token at its group's true **time centre**; a stride > 1 is force-disabled with a warning if the Motion Context layout patch is unavailable.
- **Recompose sampler + scheduler** — new optional `audio_recompose_sampler` input (unwired, the recompose inherits the main sampler — usually the 6-step turbo pick; a dedicated `dpmpp_2m` / `res_multistep` / `dpmpp_2m_sde` converges much better at 20-40 steps). `CLSSH3AudioConfig.audio_recompose_scheduler` picks the sigma spacing: `beta` (default — measured zero loop-guard re-rolls and the smoothest seams), `linspace`, `karras` (warned against: collapses ~29/40 steps into the sigma<0.1 polish band), `exponential`, `simple`. `audio_recompose_steps` max is now 60.
- **Scene switches reset the audio anchors** — a scene boundary no longer locks the new scene's loudness/timbre/groove targets to the first scene's (that dragged every later scene's music back to scene 1); the waveform anchor still tracks the previous chunk, so level continuity across the cut is kept.
- **Workflow sync** — every bundled workflow carries the new required `CLSSH3AudioConfig` keys (a prompt missing one fails validation; verified against the live ComfyUI validation path).

**2026-09-28 — Spectrum forecast: DiT step skipping for turbo runs, step-cache upgrade**

- **`CLSSH3SpectrumForecast`** — training-free hidden-state forecasting adapted to the CLSS loop from [xmarre's ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) (mechanics cross-checked against that port; nothing vendored), itself a MiniMax H3 port of **Spectrum** — *Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration* by Han et al. ([paper](https://arxiv.org/abs/2603.01623)). Wire it between the model loader and the CLSS guider. On **actual** steps the packed target-stream hidden state is captured right after the last DiT block (CPU-parked anchors); on **forecast** steps every DiT block is bypassed and a Chebyshev/ridge or plain linear extrapolation of the anchors is injected straight into the hidden stream before the native FinalLayer — per-step adaln modulation, denoise-mask rows and unpatchify still run with the exact current-sigma parameters, so only the transformer stack (~99% of step cost) is skipped. Policy per sampling pass: the first `warmup_steps` (2) and last `tail_actual_steps` (1) calls always run for real; `pattern=alternate` (default) forecasts every second step in between — a 6-step turbo chunk runs A A F A F A (~50% of DiT evals skipped per chunk); `consecutive` is the more aggressive classic A A F F F A. Fail-closed: another plugin's DiT block replacements disable forecasting for that model, a wild fit or shape mismatch re-runs the step for real, and a failed anchor capture leaves the model plain.
- **`step_cache_thresh` upgraded** — a skipped step no longer reuses the stale prediction: it is Taylor-extrapolated along the sigma axis (first-order drift of the two last computed steps), and a **per-tile veto** cancels the skip when the worst 10% of latent tiles moved more than 2.5× the threshold, so a calm background can no longer drag a moving subject through a skip. The first two computed steps of each pass warm up the derivative. When Spectrum is armed on the model the output-level step cache disables itself for it — both skip DiT evals, but Spectrum keeps the per-step modulation exact.

**2026-09-27 — video references on the R2V scene-ref nodes**

- **Video refs (`<Video k>`)** — `CLSSH3SceneReferences` and `CLSSH3SceneReferencesAll` gained `ref_videos` (AutoGrow, up to 3; VIDEO or IMAGE sockets) plus index-paired `ref_video_audios` (AUDIO or VIDEO): frames resample to 24 fps when the source differs, the video's own soundtrack auto-attaches as its audio block (an explicit `ref_video_audio` overrides; without `audio_vae` the soundtrack is skipped with a warning while the video still rides), and the prompt references it as `<Video k>`. Presentation follows the stock ref2va path (one vision block per 2 frames at true timestamps, frames snapped to the 17k+5 grid, never upscaled); a soundtrack's `<Audio j>` block rides with the video on the same scene.

**2026-09-25 — attention override, base refine guider, speed knobs, loop-guard drafts, hardening**

- **`CLSSH3AttentionOverride`** — run the DiT on **SageAttention** / **FlashAttention 2** / **xformers** through ComfyUI's per-model `optimized_attention_override` hook (stock pytorch / sub-quad included for A/B). The backend is smoke-tested on your GPU once; a missing package or a failing call falls back to the stock attention with a log warning — never a crash. Typical measured win for SageAttention on Ampere/Ada: ~1.3–1.8× on the attention part of each step.
- **`CLSSH3BaseRefineGuider`** — the audio recompose pass on **base weights without a second checkpoint**: clones the turbo model and strips its LoRA weight patches in a shared-weight copy (ComfyUI re-patches in place on the pass swap; the sampler skips its between-chunks unload for it). Wire it to `audio_refine_guider` in place of the second `CLSSH3Guider`.
- **Experimental speed knobs, both default off** — `CLSSH3Config.overlap_evict_after` (two-phase overlap eviction: a continuation chunk runs the first fraction of its steps on the full window, then drops all but the last 2 overlap rows and continues on the shorter window, re-noised from the intermediate x0 with corrected positional times; auto-disabled without the included layout patch) and `CLSSH3Config.step_cache_thresh` (skip the whole DiT forward when the latent has barely moved since the last computed step — ~0.06 typically skips 1–3 steps per chunk; never the first step, the low-sigma tail or two steps in a row).
- **Loop-guard drafts** — re-roll attempts first run a cheap draft (`CLSSH3AudioConfig.loop_guard_draft_frac`, default 0.4): a draft that already measures as a vamp is abandoned early; a good draft continues on the rest of the schedule from its own x0, so an accepted attempt still costs exactly one take.
- **Hardening** — the seam crossfade snapshots the original seam frames (no progressive smearing across a multi-junction file); the CPU noise field is sized to the run instead of always the 40 k-token cap (several GB of RAM saved); a window shorter than the overlap drops only what it holds (audio no longer slides ahead of the video); scene switches no longer lock a scene's overlap contrast to the first scene's; both `PYTORCH_ALLOC_CONF` spellings are honored.

**2026-09-23 — 16 GB memory hardening (unload before sampling, expandable segments)**

- Every sampler run calls `unload_all_models` + a cache flush **before the DiT load** and prints the freed GB (`[CLSS] unloaded all models before sampling (X GB freed)`): a resident-mode ClipProj text encoder is pinned and ComfyUI's own eviction can't touch it, but the pack's unload hook releases it for real (measured: 10.23 GB freed; the 8B-encoder OOM was 10.23 GB pinned + the DiT). No extra node in the graphs.
- The sampler enables CUDA `expandable_segments` at runtime when the environment doesn't and prints the allocator state. A 0.8 MP continuation OOM measured as allocator fragmentation (4.26 GiB reserved-but-unallocated against a 4.23 GiB request) while same-size fresh runs completed. For a clean start: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before launching ComfyUI.

**2026-09-22 — continue & re-edit a saved run**

- **`CLSSH3ContinueFromVideo`** — keep generating a finished run: reads its saved frames + audio by prefix, turns the last overlap span into the opening context (keyframe replay + SLB seed + audio tail ref) and delivers only the new span. The 5-px head the token grid needs is folded into the decoded stream and dropped at save (first 5 frames + 8.67 af of audio) — nothing to trim by hand.
- **`CLSSH3ReeditChunk`** — rebuild ONE chunk at its original span (head included): context from the saved frames before it, first/last frames pinned from the saved pixels, optional strided video ref of the saved window; the node reports `start_frame` / `frames` so the re-render replaces exactly that span in the saved sequence.
- Workflows: [`continue_minimaxh3_clss.json`](workflow/continue_minimaxh3_clss.json) / [`reedit_minimaxh3_clss.json`](workflow/reedit_minimaxh3_clss.json) — both nodes need only the saved `filename_prefix` / `audio_prefix`, the same template, overlap and VAE(s).

**2026-09-20 — audio loop guard + a dedicated audio config node**

- **`CLSSH3AudioConfig`** — all audio settings moved off the sampler onto one node (recompose steps/σ/arc margin/pool/stride/seed/ref span, head discard, loop guard); wire it into the sampler's `audio_config`. The sampler's own audio widgets were removed.
- **Loop guard** — after each recompose take the delivered span is measured for within-chunk repetition (`aud_wc`) and cross-chunk looping (`aud_loop`); takes that measure as a vamp are re-rolled with the next seed (up to `loop_guard_rerolls`, default 2), optionally rebuilding the recompose ref at `loop_guard_retry_ref_ms` (default 2000 ms, the ear-validated rescue). The least-repetitive take is kept — the original survives unless a re-roll measures better. Skipped on scenes whose audio rides their own `<Audio j>` reference.

**2026-09-17 — scene-grid ref-audio windows (drift fix, automatic)**

- **`CLSSH3SceneReferencesAll` ref-audio windows are cropped AUTOMATICALLY to each scene's exact span** — the node encodes every window with a ±4 s guard band and stashes the pristine windows; the sampler crops them at run start to the exact delivered spans (from its own chunk plan + scene allocation; **no VAE, no geometry, nothing to wire** — a span outside the guard falls back to a re-encode when an `audio_vae` is wired and is logged otherwise). The guard absorbs the old exact-T lattice slip (83.3 ms per chunk on the 17k+5 grid; `sim/sim_ref_window_drift.py`). The sampler also warns if a window length still doesn't match its scene span.

**2026-09-12 — sampler option cleanup**

- **Removed sampler options** — `refine_latent` (the two-pass refine), `audio_cfg_cont` (the guider's `audio_cfg` now applies to every chunk), `audio_refresh_waveform` and the sampler's `audio_vae` input (the audio continuity reference is always the delivered latent — the VAE round-trip was lossy and compounded). Dead keys dropped from the canonical workflows.

**2026-09-11 — all-scene refs, global prompt text, audio recompose + seam controls**

- **Chunk-by-chunk upscaling inside the sampler** — `CLSSH3LoadLatentUpscaleModel` + the sampler's `upscaler` / `upscale_scale`: every chunk is upscaled **after its SLB step** (full window in, overlap cross-faded over the previous tail), so a long video never exists at high res at once — the per-chunk, memory-bounded version of [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler). See [Upscaling](#upscaling-neural-latent-upscaler). The sampler also accepts **any slice** of the 1.0→0.0 flow schedule (a low-res pass may end above zero; a partial schedule may also start below 1.0).
- **All-scene R2V refs** — new `CLSSH3SceneReferencesAll` node: every `ref_image` attaches to ALL scenes, and the connected `ref_audio` file(s) are concatenated and cut into consecutive per-scene windows (`audio_seconds_per_scene`, default 10 s → "10 s after 10 s"). Replaces chaining one ref node per `---` block; [`ref2v_minimaxh3_clss.json`](workflow/ref2v_minimaxh3_clss.json) uses it.
- **`global_text` on `CLSSH3ScenePrompts`** — one text field copied to the top of every scene block before encoding, so shared style/section text is written once; the prefix is baked into each scene's text and survives the ref nodes' re-tokenization.
- **Audio recompose** — `audio_recompose_steps` / `_sigma` / `_pool` / `_stride` / `_seed` + `audio_refine_guider`: per chunk, a fresh audio take from pure noise against the finished video (the measured fix for turbo-LoRA audio). Off in the canonical workflows; `sim/make_audio_lab.py --recompose` drives it.
- **Audio seam controls** — the `ref_audio` continuity block is now placed through the H3 layout patch so it **ends exactly at the join** (length and end position decoupled); `audio_head_discard_ms` drops the unstable opening of continuation chunks; `audio_xfade_ms` crossfades the join.
- **Determinism** — cross-chunk noise fields are generated at fixed caps, so chunk 1 is bit-identical regardless of `num_chunks` (`torch.randn` has no prefix property).
- **Cleanup** — dead guide-layout check removed; the long inline design notes in `nodes.py` trimmed down to the measured facts (details live in this README and `AGENTS.md`).

## Support

If this node pack is useful to you, you can support its development on Patreon: **[patreon.com/c/AleksanderM](https://www.patreon.com/c/AleksanderM)**

## Acknowledgements

Built on [MiniMax H3](https://huggingface.co/Comfy-Org/MiniMax-H3) by MiniMax (weights under the MiniMax H3 Community License — read it before commercial use), [LTX-2](https://github.com/Lightricks/LTX-2) by Lightricks, and the ComfyUI ecosystem.

The `CLSSH3SpectrumForecast` node adapts **Spectrum** — *Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration* by Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo and Stefano Ermon ([paper](https://arxiv.org/abs/2603.01623), [official implementation](https://github.com/hanjq17/Spectrum)) — to the CLSS streaming loop, with the mechanics cross-checked against [xmarre](https://github.com/xmarre)'s ComfyUI MiniMax H3 port: [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3).
