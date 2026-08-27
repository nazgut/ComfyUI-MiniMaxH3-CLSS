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
- **Memory reality on 16 GB:** the full qwen3vl-32B text encoder (15.7 GB) plus the
  int8 DiT (~12 GB) plus full-resolution activations do NOT fit together — the
  canonical workflow therefore uses the ClipProj pack's `ClipProjLoader`
  (Qwen3-VL-4B + learned projection, ~5.5 GB, API-compatible: `tokenize` /
  `encode_from_tokens_scheduled` work unchanged) and 832×480 windows
  (~28k packed tokens/step). The validated working reference stack (user's):
  int8 convrot DiT, MiniMaxH3SigmaShift 12.0/6.0, 832×480, 243 px windows, 12 steps.

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
UNETLoader / ClipProjLoader (Qwen3-VL-4B + projection) / VAELoader×2 (video int8, audio bf16)
CLSSH3ScenePrompts(CLIP, prompts)          → CONDITIONING (one entry per scene, '---' split;
                                             used twice: positive scenes + negative)
CLSSH3Guider(model, pos, neg, video_cfg 1.0, audio_cfg 1.0, rescale 0.7) → GUIDER
                                             # H3 is CFG-distilled — live A/B measured 4.0/7.0
                                             # corrupting output into oversaturated glitch frames;
                                             # 1.0/1.0 = off and skips the uncond eval entirely
MiniMaxH3SigmaShift (stock, 12.0/6.0)      → MODEL (feeds guider + scheduler)
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


# Verified upstream facts (ComfyUI master, MiniMax H3 / PR #15224)

Target file under review: `/mnt/agents/upload/user_pasted_clipboard_long_content_as_file_this is minimax h3 n.txt`
(code starts at line 3; lines 1-2 are user prose). Symptom: finished run, but decoded video AND audio are pure noise.

The following have been VERIFIED against upstream master source — do NOT re-flag these as bugs:

1. `comfy_extras.nodes_custom_sampler.SamplerCustomAdvanced` is an `io.ComfyNode` with `@classmethod execute(cls, noise, guider, sampler, sigmas, latent_image)` AND the alias `sample = execute`. Calling `SamplerCustomAdvanced().sample(...)` works (classmethod). It returns `io.NodeOutput(out, out_denoised)`; `io.NodeOutput` defines `__getitem__` (self.args[i]) so tuple-unpacking `_, denoised = ...` works via the legacy sequence protocol. `out_denoised["samples"]` = x0 from the preview callback, process_latent_out'ed; for nested latents the callback hands x0 as an already-unpacked NestedTensor.
2. `comfy.samplers.calc_cond_batch(model, conds: list[list[dict]], x_in, timestep, model_options)` returns a Python LIST of tensors (one per cond entry) — `[0]` unwrap is correct (stock Guider_DualModel does exactly this).
3. `comfy.utils.pack_latents(list_of_tensors)` returns a TUPLE (packed [B,1,N], latent_shapes) — `[0]` indexing is correct. `unpack_latents(packed, shapes)` returns a list.
4. `comfy.nested_tensor.NestedTensor` — constructor takes iterable, `.unbind()` returns the underlying list, `.is_nested` attribute exists, `.to()/.cpu()/.float()` map over members.
5. denoise_mask semantics: 1 = regenerate/denoise, 0 = preserve. KSamplerX0Inpaint (samplers.py L630-643) + MiniMaxH3.scale_latent_inpaint (model_base.py L2248-2272) implement cond-strength re-injection (video aug 0.999, audio 1.0); per-row sigma = m·sigma_stream happens inside the DiT (model.py L587-609), clamped at cond pin.
6. Nested noise_mask IS supported: CFGGuider.sample (L1297-1314) unbinds a nested mask, runs prepare_mask per stream, repacks via pack_latents. Video mask [1,1,T,1,1] -> trilinear interp to (T,H,W); audio mask [1,1,2,Ta] -> bilinear to (2,Ta) (dims=2 path: reshape (-1,1,2,Ta)).
7. CFGGuider.sample packs nested latent+noise to [B,1,N] via pack_latents BEFORE the sampler runs, and sets latent_shapes; inner_sample does process_latent_in only if latent nonzero; output is process_latent_out'ed in packed space and re-nested at the end. MiniMaxH3 process_latent_in/out scale/unscale ONLY the audio slice by audio_scale = shift/audio_shift (=4.0 for 12/3).
8. Conditioning: node-facing CONDITIONING entries are PAIRS [tensor, dict] (conditioning_set_values uses t[0]/t[1]; MiniMaxH3AddGuide uses positive[0][1]). BUT guider.original_conds entries are DICTS after inner_set_conds -> convert_cond ({"cross_attn": tensor, "model_conds":..., "uuid":...}). So treating original_conds["positive"][i] as a dict with .get("cross_attn") and {**entry, "minimax_keyframes": ...} is CORRECT.
9. Keyframe conditioning format verified: {"resolved_frame_index": int (pixel frames, 0-based, window-relative), "latent": video-VAE latent (single image -> [1,24,1,H/16,W/16]), optional "audio_latent"}. Key "minimax_keyframes" = list of such dicts. consumed at cond_t = cursor + FRAME_RESCALE*resolved_frame_index.
10. Latent shapes: video [B,24,T,H/16,W/16], audio [B,32,2,Ta]; NestedTensor((video, audio)). Constants: FPS=24, AUDIO_LATENT_FPS=40, FRAME_PER_TOKEN=(1,4,4,4,4), FRAME_RESCALE=5/3, VISUAL_COND_TIMESTEP=0.999, AUDIO_COND_TIMESTEP=1.0. temporal_shape: frame_count snapped UP to 17k+5; video_latent_t = 2+5k; audio_t = round(frames/24*40).
11. Video VAE: vae.decode(video_latent) returns [B, T_px, H, W, 3] in [0,1] (wrapper movedim(1,-1)). first_stage_model HAS decode_output_shape(input_shape)->(B,3,T_px,H*16,W*16); vae_ratio_t=4 with the (1,4,4,4,4) pattern anchored at the slice start (standalone n-token decode = 17k+5-style count). Audio VAE: vae.decode([B,32,2,Ta]) -> movedim(-1,1) -> [B,2,L]; std*5 normalize floored at 1.0; sample rate via audio_sample_rate_output -> audio_sample_rate (=32000 for H3) -> 44100.
12. MiniMaxH3SigmaShift patches model_sampling (ModelSamplingAV+CONST, shift/audio_shift) + transformer_options keys; stock graph uses BasicGuider (no CFG) at shift 12/3.
13. CFGGuider.set_conds(positive, negative) / set_cfg exist. copy.copy(guider) + reassigning .original_conds works: sample() rebuilds self.conds from original_conds each call, and outer_sample re-prepares self.inner_model per call.
14. k-diffusion samplers receive model_k = KSamplerX0Inpaint wrapping the guider; predict_noise override on a CFGGuider subclass IS the hook that gets called (guider.__call__ -> outer_predict_noise -> self.predict_noise).
15. In CFGGuider.sample the first branch is `if sigmas.shape[-1] == 0: return latent_image`.

So the noise bug is most likely in the node file's OWN logic (grid math, slicing, masks, SLB bookkeeping, assembly, or the guider/noise glue), or in how it uses the vendored clss.py (CLSSConfig/CLSSState: overlap_latent, post_process, update_buffer, top_anchors, reset_drift_refs — clss.py is NOT available, assume its API behaves as the names imply).

Known minor anomalies already found (judge whether they matter):
- Per-chunk audio window length = Ta_ol + cur_new_af from CUMULATIVE rounding can differ by +/-1 audio frame from round(window_px * 5/3) (e.g. 179 vs 178 for a 22+85px window with overlap 7).
- _px_for_tokens fallback `return 4 * n` is wrong math but dead code when decode_output_shape exists (it does).
- INPUT_TYPES default context_frames=0 but decode_save signature default is context_frames=2.

Report: the most plausible root cause(s) of pure-noise output on BOTH streams, ranked, with exact line numbers and a concrete fix for each.
