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

- **Calibrated context re-noising** (`tau_c`, default 0.05; per-chunk schedule rises
  toward a 0.10 ceiling with a 5-chunk half-life) — on H3 implemented via per-token
  denoise masks (mask m → per-row sigma m·σ, `comfy/ldm/minimax/model.py` `_forward`)
- **EMA-tracked per-channel AdaIN drift correction** (`beta`, default 0.4) with an
  anchored, capped mean (`ema_mean_max`); the EMA reference resets at every scene
  change via `CLSSState.reset_drift_refs`
- **Keyframe context replay** — the previous chunk's context rows ride as
  `minimax_keyframes` conditioning rows at their pixel times (Motion Director
  mechanism); the VIDEO span is additionally seeded in the latent and re-noised
  at τc. The AUDIO window head stays FREE — only the end-anchored
  `minimax_refs` audio ref carries the tail (MD parity since 2026-09-16; the
  old τ_a in-stream seed is deleted). With the sampler's optional `audio_vae`
  input wired, that ref is REFRESHED AT EVERY BOUNDARY from the delivered tail
  re-encoded in the export (audible) domain — MD `audio_context_refresh`
  normal path; the delivered latent slice is the strict fallback (MD
  `_waveform_can_refresh`; the per-chunk log marks the source `(wav)`/`(lat)`).
  All workflow JSONs carry the wiring. OWNER DIRECTIVE 2026-09-18: "no tail
  in audio when there is a ref" — on chunks whose scene carries its OWN
  audio reference (`<Audio j>` R2V block) the tail ref is NOT attached at
  all (`audref=off(scene ref)` in the chunk log); those scenes take their
  audio FROM the reference. The VIDEO continuation — keyframe replay + SLB
  seed — runs its overlap everywhere, refs or not.
- Two-band spatial detail anchor (`detail_anchor` input)

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
nodes.py     # all 12 ComfyUI node implementations (incl. R2V scene-reference nodes:
             # CLSSH3SceneReference single + CLSSH3SceneReferences V3-Autogrow multi,
             # re-tokenizing one scene's text with minimax_ref_items so <Picture N>/
             # <Audio N> labels bind per scene; CLSSH3SceneReferencesAll applies
             # images to every scene and encodes ref audio into per-scene GUARDED
             # windows (±4 s band) — the sampler re-cuts each to the piece's
             # scene span PLUS the window lead-in, at run start, by CROPPING the
             # guard (PRESENTATION LEAD-IN, 2026-09-18: H3 presents a scene's
             # <Audio j> ref from the ref's own beginning == the window's start
             # while delivery begins one overlap later — the window starts
             # FRAME_RESCALE·px_ol af early (0 for the piece's first scene);
             # without it the first ~0.92 s of every continuation scene was
             # swallowed — measured +916.7/+925 ms shifts, NCC 0.91 at the
             # offset vs 0.02 aligned; crops 405/434/433 af on the 3×243px plan)
             # (_recut_scene_audio_windows + _scene_grid_window_bounds, from its
             # own chunk plan + scene allocation; nothing to wire on the sampler
             # side, nothing to enter by hand). A span outside the guard falls
             # back to a re-encode (needs the sampler's audio_vae) and is logged
             # otherwise — the old exact-T cut slid ~83 ms per chunk against the
             # 17k+5 grid (the sampler warns when window af lengths do not match
             # its scene spans + lead-in; sim/sim_ref_window_drift.py); and
             # REF MERGE ORDER CONTRACT (2026-09-18): the sampler's continuation
             # tail ref is APPENDED LAST to minimax_refs, after the scene's own
             # blocks. Upstream binds <Picture i>/<Audio j> ordinals BY ORDER
             # among same-kind blocks, so prepending the tail stole the scene's
             # <Audio 1> slot — the previous tail got re-presented in the next
             # chunk and the scene refs stopped binding after chunk 1 (measured:
             # chunk1 NCC 0.90 vs the track, chunks 2/3 ~0). MD parity
             # (motion_context.py: existing_refs + [block]);
             # guard: sim/sim_ref_order.py. NEVER prepend a ref. AND the tail
             # ref is attached ONLY on chunks whose scene has NO audio ref
             # (owner directive 2026-09-18) — ref'd scenes take their audio
             # from <Audio j>; the video continuation is unaffected.)
             # CLSSH3ContinueFromVideo + CLSSH3ReeditChunk (RUN CONTINUATION /
             # RE-EDIT): re-encode a SAVED run's own frames (its decode-save
             # prefix under output/) into a CLSS_CONTEXT for the sampler — the
             # last overlap frames + the audio tail for continue, or chunk k's
             # surrounding frames (+ first/last frame pins from the saved
             # pixels, the overlap audio before the chunk, and an optional
             # strided video ref of the saved window) for re-edit. With a
             # context the sampler OPENS ON A CONTINUATION WINDOW: context rows
             # keyframed AND seeded at tau_v, the saved audio as the tail ref,
             # delivery = the new span only (continue, see the HEAD RULE and
             # HEAD TRIM below) / the chunk's span including the head (re-edit,
             # so it drops back into the saved video).
             # HEAD RULE (2026-09-22): a context run has no piece head, so the
             # first delivered window re-covers the saved run's LAST 2 CONTEXT
             # TOKENS (5 px / 0.21 s) as its head and the audio join cut moves
             # to px ov-5. The model maps window tokens with (1,4,4,4,4)
             # anchored at the WINDOW start (model.py _video_t_grid), the VAE
             # decodes the assembled latent from the PIECE start
             # (decode_output_shape: 72 tok -> 243 px); tokens ov-2/ov-1 carry
             # model spans (1,4) - exactly the decode spans of assembled
             # positions 0/1 - and every later token then agrees (window phase
             # ov%5 == the assembled phase 2 that the 72-token first window
             # establishes). Without the head every 5th delivered token renders
             # at the wrong span (4->1 then 1->4) = a 17-frame stutter/freeze
             # cycle. So chunk 1 delivers 243 px (head + 238 new) and chunks
             # 2+ deliver 238 px; a re-edit span is [boundary-5, boundary+238)
             # and the pins sit at its edges (window px 17 / 259). HEAD TRIM
             # (owner report 2026-09-22: "now we have repeated last 5 images"):
             # the head stays in the DECODED stream (the phase needs it) but the
             # CONTINUE output must not repeat it, so CLSSH3ContinueFromVideo
             # marks the context trim_head=True and decode-save drops the first
             # 5 frames plus the first 8.67 af (head_af + 5 px = 6933 smp; the
             # audio is then padded/truncated to the trimmed span's law) — the
             # saved take starts at the new span, nothing to trim by hand.
             # CLSSH3ReeditChunk sends trim_head=False: its head IS the span it
             # replaces. The plan is
             # bit-identical to the frozen implementation for the no-context
             # path; sim/sim_continue.py pins the plan table, the
             # spans/pins, slice math and the on-disk naming (frames are
             # written as <prefix-basename>_<idx:05d>.png, i.e. a prefix that
             # already ends in '_' yields a DOUBLE underscore). Workflows:
             # workflow/continue_minimaxh3_clss.json + reedit_minimaxh3_clss.json.
             # CLSSH3LoadLatentUpscaleModel + the sampler's per-chunk upscaler,
             # which SOFT-IMPORT their model module by file path at execute time
             # from sibling custom_nodes/*/nodes/minimax_h3_latent_upscaler_3d.py
             # — nothing vendored, no import-time code loading)
clss.py      # model-agnostic CLSS core: CLSSConfig, CLSSState (SLB, EMA/AdaIN
             # drift correction, post_process, reset_drift_refs) — no ltx imports
__init__.py  # node-mapping exports only
workflow/    # canonical workflows (API format): t2v_minimaxh3_clss.json (text-to-video),
             # i2v_minimaxh3_clss.json (first-frame guide via the sampler's image/vae
             # inputs), ref2v_minimaxh3_clss.json (CLSSH3SceneReferencesAll R2V refs),
             # continue_minimaxh3_clss.json (keep generating a finished run: context =
             # the saved run's own tail) and reedit_minimaxh3_clss.json (re-render ONE
             # chunk of a finished run: context + first/last frame pins + optional
             # video ref; built from the SAME graph as continue - the owner's turbo R2V
             # production stack, only node 33 swapped and the output prefixes moved to
             # clss_h3/reedit/ + audio/reedit/). EVERY sampler run first calls
             # `_unload_before_sampling` (mm.unload_all_models + soft_empty_cache,
             # printed with the GB freed) BEFORE the first DiT load: a resident-mode
             # ClipProj text encoder is pinned and ComfyUI's own eviction no-ops for
             # it, but the pack hooks `unload_all_models()` to release the pins for
             # real (its log: "... unloaded (10.23 GB freed)") - so the encoder
             # leaves the card right after the text encode instead of OOMing the
             # DiT load (owner OOM 2026-09-22 with the 8B encoder: 10.23 GB pinned
             # + DiT > 15.6 GB). No extra node in the graphs. The sampler also
             # enables CUDA "expandable_segments" at runtime when
             # PYTORCH_CUDA_ALLOC_CONF lacks it and prints the allocator state
             # (`_enable_expandable_segments`): the 0.8 MP CONTINUITY OOM
             # (2026-09-23) was fragmentation (4.26 GiB reserved-but-unallocated
             # vs a 4.23 GiB request) while same-size FRESH runs completed -
             # continuity's packed sequence is ~15% longer (81-token window =
             # 7 ctx + 70 new + 4 margin = 273 px, plus 7 keyframe frames; fresh =
             # 76 tok / 256 px, no keyframes). Full fix = the env var at start.
             # t2v/i2v/ref2v stay on the validated
             # stack: 832x480, 243 px windows, 20 steps, shift 12/6, audio_cfg 4,
             # tail_margin 12, audio_config wired (10 chunks for t2v/i2v/ref2v, 1 for
             # continue/reedit as saved).
             # The audio-lab files are generated on demand: sim/make_audio_lab.py
             # writes workflow/audio_lab_minimaxh3_clss.json (driver: t2v).
             # RULE: every experiment copies the canonical file — never mutate it in place.
sim/         # offline harnesses (no model): sim_tau_c, sim_geometry, sim_mean_drift,
             # sim_splice, sim_audio_refresh, sim_preunload, sim_ref_window_drift
             # (per-scene ref-audio windows vs the piece's scene grid) +
             # sim_continue (continuation/re-edit context: plan table vs the frozen
             # no-context implementation, grid identity that forbids the 17k+5 head
             # on a context window, delivery/cut/splice arithmetic, re-edit spans +
             # pins, video-ref clip length, audio slice law, frame-file loading) +
             # sim_video_ref_span (Motion Video Context: a strided recompose video
             # ref must sit on the target time grid — nodes._mc_fixup_video) +
             # sim_loop_guard (a recompose take that measures as a self-repeating
             # vamp is re-rolled with the next seed — _take_loop_metrics,
             # _loop_guard_bad/_pick, _rc_seed_for) +
             # sim_audio_continuity (CLSSH3ScenePrompts.audio_continuity_text:
             # off-path byte-identity, dual-encode + _apply_scene_cont swap
             # gated on chunk 0 / ref'd scenes; make_audio_lab --continuity-text) +
             # audio_forensics.py (FLAC analyzer: loop/dulling/join metrics) +
             # audio_repeat_scan.py (waveform repeat scan: exact period/NCC/reldiff) +
             # make_audio_lab.py -> audio_lab workflow (see sim/AUDIO_LAB.md)
```

### Relationship to the LTX repo

The LTX repo owns the LTX-2.3 implementation and its own `Ltx-2-CLSS` submodule. This
repo's `clss.py` was vendored from that submodule (LTX-specific conditioning methods
removed; the `overlap_latent` accessor added). If the algorithm core changes
materially, port the change both ways by hand — there is no shared dependency.

## The nodes and how they wire together

```
UNETLoader / ClipProjLoader (Qwen3-VL-4B + projection) / VAELoader×2 (video int8, audio bf16)
CLSSH3ScenePrompts(CLIP, prompts)          → CONDITIONING (one entry per scene, '---' split;
                                             optional global_text copied to the TOP of every
                                             scene block before encoding; optional
                                             audio_continuity_text encoded as a SECOND text
                                             variant per scene ('clss_cont') and swapped in
                                             ONLY on chunks carrying the continuation tail
                                             ref (never the run's first chunk; _apply_scene_
                                             cont, owner directive 2026-09-21); test branch
                                             experiment/audio-continuity, default empty =
                                             off; used twice: positive + negative)
CLSSH3Guider(model, pos, neg, video_cfg 1.0, audio_cfg 1.0, rescale 0.7) → GUIDER
                                             # H3 is CFG-distilled — live A/B measured 4.0/7.0
                                             # corrupting output into oversaturated glitch frames;
                                             # 1.0/1.0 = off and skips the uncond eval entirely
MiniMaxH3SigmaShift (stock, 12.0/6.0)      → MODEL (feeds guider + scheduler)
EmptyMiniMaxH3LatentAV (stock)             → LATENT (per-chunk AV template, 5k+2 grid)
CLSSH3Config                               → CLSS_CONFIG
CLSSH3AudioConfig                          → CLSS_AUDIO_CONFIG (ALL audio settings:
                                             recompose steps/sigma/arc margin/pool/stride/
                                             seed/ref span, head discard + the loop guard &
                                             rescue ref span; wire into the sampler's
                                             `audio_config` — the sampler's own audio_*
                                             widgets were removed 2026-09-20)
CLSSH3LoadLatentUpscaleModel               → LATENT_UPSCALER (optional; soft-imports
                                             the sibling upscaler pack's 3D module)
KSamplerSelect + BasicScheduler + RandomNoise → SAMPLER / SIGMAS / NOISE
CLSSH3ContinueFromVideo / CLSSH3ReeditChunk  → CLSS_CONTEXT (reads the SAVED run's
                                             frames + audio from output/ by prefix;
                                             wire into the sampler's
                                             `continuation_context` — the run then
                                             opens on a continuation window)
CLSSH3StreamingSampler(...)                → LATENT  (chunked, full telemetry;
                                             optional `continuation_context` =
                                             continue/re-edit opening context;
                                             optional upscaler = per-chunk neural
                                             upscale after each SLB step; any
                                             slice of the 1.0→0.0 schedule)
CLSSH3VideoDecodeSave(vae, audio_vae, ...) → PNG frames on disk + AUDIO
```

**Per-chunk upscale (in-sampler):** `upscaler` + `upscale_scale` (1.0–4.0) run the
neural upscaler on EVERY chunk right after `clss_state.update_buffer(corrected)`
— the streaming state (SLB, corrections, telemetry) stays low-res; only the
delivered pixels grow. Input is the full window (`vid_out[:, :, :ov]` +
`corrected`), output is appended minus the overlap tokens, and the overlap span
is cross-faded over the previous chunk's delivered tail by
`_blend_upscaled_overlap` (ramp 0→1; time is preserved by the upscaler, so token
indices line up). Output latent = high-res video + unchanged audio; the model is
moved to the **compute device** (`comfy.model_management.get_torch_device()`) at
run start and offloaded at the end — NEVER to the latent's device: under
ComfyUI's dynamic VRAM loading the AV template lives on CPU and fp16 inference
on CPU is pathologically slow (measured: a 5-frame 8×8 latent took 205 s). The
wrapper (`_H3UpscalerHandle.upscale`) mirrors the upscaler node's execute:
per-channel H3 normalisation, `target_size=(T, h', w')`, `enable_chunking=True`,
32-px canvas alignment (→ even latent dims); on a CPU-only machine it falls back
to fp32. Per-chunk upscale time is printed (`[CLSS]   upscale N tok -> AxB px in
Xs`).

**Partial schedules:** the sampler accepts ANY slice of the 1.0→0.0 flow
schedule — a low-res pass may end above 0 (its x0 is then what the upscaler
carries), and a partial schedule may start below 1.0 (every chunk then starts
from noise at σ0).

**Sample-exact audio seams:** the sampler cuts each delivered slice at
floor(px_ol·5/3) audio tokens (22 px → 36 af, while the exact boundary is
36.667 af) and records the junction geometry in the latent
(`clss_audio_splice`). `CLSSH3VideoDecodeSave` rebuilds the waveform from it —
drops the duplicated head samples, moves the next slice's shortfall fill
across the junction, zero-pads ≤ 1 audio step at the tail — so joins are
sample-contiguous (the old whole-token cut carried a 16.7 ms duplicate/skip
at EVERY seam; `sim/sim_splice.py` verifies, `_splice_delivered_audio`).
Junctions fall back to the plain append when the head is discarded
(`audio_head_discard_ms`) or the window is degenerate. The seam carries
NOTHING else — Motion Director parity (2026-09-16): no junction lead, no
handoff fade, no join glide, no attack tame, no chunk level
match (all removed; MD's seam is an exact trim + plain `torch.cat` and its
ref ends exactly at the join in both content and row placement — see the
SEAM DOCTRINE note in `nodes.py`). ONE measured exception: each junction is
a take-swap CROSSFADE over the h−f samples both takes re-rendered (8–17 ms)
— the hard swap measured 0.16–0.24 amplitude (4–5× the local median |diff|)
in lab_00032, an audible click; a CONTENT fix (both sides are the same
wall-clock render), never a gain. The decode stage applies ZERO level/dB
processing (owner directive 2026-09-17: "we should not manipulate audio at
all in correcting db"): the junction-anchored level matcher and its 6 dB
content gate were deleted after the matcher measured ducking the owner's
ref track's own section change by −16.7 dB (CLSSH3_00002) — the only
remaining gain touch in the audio path is the headroom guard (whole-file
attenuation only when the take would hard-clip the 16-bit FLAC).
`audio_xfade_ms` / `audio_join_lead_ms`
are deprecated + ignored so old workflow JSONs still load.

**Tail margin (`tail_margin_px`, default 12):** MD generates every segment on
the 17k+5 grid UP from `target+context` and EXPORTS only `[context,
context+target)` — the 1-16 px alignment surplus is generated and discarded
(their 124-px default segment discards 12 px; a 10-s segment, 15 px). The
discarded zone is where each take's end-of-generation wind-down sits — the
measured 150-250 ms quiet dip at the window end of our files — and MD's next
ref ends at the EXPORTED end, so the dip never enters the chain. Our windows
are grid-exact (238+22=260 px), surplus 0, so we delivered the dip into every
seam (measured -10/-12 dB joins). With the margin > 0 every chunk GENERATES N
px past its delivery end (token run on the (1,4,4,4,4) pattern; 12 -> 4 tokens
= 13 px = 22 af) and delivers only up to it; the SLB (`update_buffer` gets
`corrected` = the delivered slice), the next chunk's keyframes, the audio tail
ref, the decode splice (`delivered_af`) and the assembled latent all see the
PRE-margin material — MD's discarded surplus, made deterministic. The soft
12-s window cap counts the margin; `0` restores the old behaviour.

`scene_handoff`: `transition_chunk` (default) = two-step crossfade straddling each
boundary (outgoing block's last chunk 25%-incoming, incoming block's first chunk
75%-incoming; needs every scene block ≥2 chunks, i.e. `num_chunks ≥ 2×scenes`);
`blend` = single 50/50 chunk; `hard` = plain text swap. EMA/refs reset on the first
incoming-leaning chunk.

Audio settings live on `CLSSH3AudioConfig` since 2026-09-20 (recompose
steps/sigma/arc margin/pool/stride/seed/ref span, head discard, and the loop
guard: `loop_guard_rerolls` 2 / `loop_guard_wc` 0.99 / `loop_guard_loop` 0.70
/ `loop_guard_retry_ref_ms` 2000 — the ear-validated rescue config; see
JUSTIFICATION §26 + §27). Notable remaining sampler knobs: `detail_anchor` on
(the two-band band-energy anchor);
`tail_margin_px` 12 (generated-but-not-delivered window tail, MD surplus
parity — see above); `fps`
forced to 24 with a warning (the 40-latent-fps audio math is hard-wired to 24).

## Continuing and re-editing a saved run

A finished run leaves everything needed to pick it up again: its PNG frames
(`<output>/clss_h3/...`, written by `CLSSH3VideoDecodeSave`) and its audio
(`<output>/audio/...`, written by the save-audio node). Two nodes turn that
material back into conditioning context, and are read from disk by prefix —
no loader nodes, no extra geometry to enter:

- `CLSSH3ContinueFromVideo` — **continue the piece.** The last overlap token
  span of frames (`_px_of_tokens(overlap, 0)` = 22 px at overlap 7) plus the
  matching audio tail become the new run's opening context, so its first
  window is a normal continuation window: the context rides as keyframe rows,
  seeds the SLB at `tau_v`, and the saved audio tail is its tail ref. The
  first delivered window carries the 2-token HEAD (the context's last 5 px,
  see the HEAD RULE above): 243 px = 5 px re-covered + 238 px new; every later
  chunk delivers 238 px (9.92 s). The HEAD TRIM (above) drops that 5 px from
  the saved output — decode-save writes only the new span (frames + audio), so
  the take appends after the saved run with nothing to trim. `num_chunks`
  picks how much to add.
- `CLSSH3ReeditChunk` — **fix one bad chunk.** Chunk k of the saved run is
  rebuilt at its original span plus the same 5 px head: chunk 1 = `[0, 243)`,
  chunk k>=2 = `[px0 + (k-2)*pxc - 5, +243)`. Chunks 2+ get the saved frames
  before them as context (rows + audio tail ref); first/last frames of the
  re-rendered span are pinned as clean keyframes (`pin_frames`, default
  first+last) so it still meets its saved neighbours; `video_ref` (default
  off) presents the saved window's own motion as a strided video ref. The
  node reports `start_frame` and `frames` (the span to replace in the saved
  sequence; the length is unchanged by the head); chunk 1 has no predecessor
  and is rebuilt from its pins alone. With `num_chunks > 1` the re-render
  simply keeps generating forward from the context — the pins anchor the
  first window.

Both nodes need the same `latent` template, `clss_config` (same overlap),
`vae` and — for the audio context — `audio_vae` as the run they continue;
their `filename_prefix` / `audio_prefix` are the exact strings the saved run's
decode-save / save-audio nodes wrote. Both also take a `scene_index` (which
prompt block the saved material belongs to — the sampler overrides the first
window's scene with it, so multi-scene runs keep the right text; single-scene
prompts leave 1). Conditions they check and report:
missing frames (the span must exist), a context row count that differs from
the run's overlap, an overlap clamped by the 12 s cap (rebuild the context),
and a `latent` template off the 5k+2 grid. The saved run must also have been
uniform (no window-cap auto-split) for the chunk index → frame index map to
hold. Everything here is generation-path adjacent: validate live before
shipping any change (sim/sim_continue.py covers the arithmetic only).

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
- Tooltips on every input; heavy docstrings citing measured evidence;
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
