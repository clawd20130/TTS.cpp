# Vulkan Backend Status

This document tracks the current state of the experimental Vulkan path for
TTS.cpp. The goal is to remove the OpenVINO dependency for GPU inference by
running TTS GGML graphs directly through ggml backends.

## Current Baseline

- This branch currently vendors a patched ggml submodule at
  `5302dc87afb20ac7ff24a4f661d89cf689884901`.
- Upstream ggml was checked at `707321c4` (`v0.15.2`) for backend coverage.
- TTS.cpp can be configured with `-DGGML_VULKAN=ON` and the `tts-cli` target
  links against `libvulkan.so.1`.
- Runtime backend selection is available through `TTS_BACKEND` or `--backend`.
  Supported values are `auto`, `gpu`, `cpu`, `metal`, and `vulkan`.
- `TTS_DEVICE` selects a backend-local device index, and
  `TTS_BACKEND_STRICT=1` disables silent fallback. Strict mode also validates
  every allocated compute graph and aborts if a non-view compute node is placed
  on CPU instead of the requested accelerator.
- The patched ggml submodule now has Vulkan kernels and backend tests for
  `EXP`, `STEP`, `ROUND`, `RECIPROCAL`, `MOD`, `CUMSUM`, `UPSCALE_LINEAR`,
  `CONV_TRANSPOSE_1D`, `STFT`, `AA_STFT`, `ISTFT`, and `AA_ISTFT`.
- A real Kokoro GGUF synthesis run now works through the Vulkan backend. The
  validation model was `Kokoro_no_espeak_Q4.gguf` with `voice=af_heart`.

## K8 iGPU Performance Target

Current K8 development target is the local AMD Radeon 780M Vulkan device
(`RADV PHOENIX`). The active Kokoro validation path is
`Kokoro_jf_alpha_no_espeak.gguf` with `voice=jf_alpha`, SBV2-provided Japanese
phonemes, and `TTS_BACKEND_STRICT=1`.

The repeatable Japanese benchmark text is:

```text
赤い血は白い雪のうえでとてもきれいに見えたので女王さまは思いました。
```

Earlier debug-frontend baseline before the generator-memory fixes:

| Backend | Median synthesis | Audio length | RTF | Clip ratio |
| --- | ---: | ---: | ---: | ---: |
| TTS.cpp CPU | `2.252s` | `5.775s` | `0.390` | `0.0` |
| TTS.cpp Vulkan / Radeon 780M | `3.360s` | `5.775s` | `0.582` | `0.0` |
| Target | `0.578s` or faster | `5.775s` | `0.100` or lower | `0.0` |

Current source-run CLI measurement on an 82-token Japanese phoneme sample after
the ConvTranspose range fix, IM2COL remap, and default LSTM gate fusion:

| Backend / mode | Duration compute | Generate compute | Total CLI time | Output length |
| --- | ---: | ---: | ---: | ---: |
| CPU | `66.600 ms` | `3212.53 ms` | `4614.65 ms` | `6.000s` |
| Vulkan default | `101.756 ms` | `1070.14 ms` | `~2.1s` | `6.000s` |
| Vulkan + `KOKORO_FUSE_ADAIN_SNAKE=1` | `101.910 ms` | `940.287 ms` | `~2.0s` | `6.000s` |

Current debug-frontend measurement through the SBV2-to-TTS.cpp proxy on the
Tailnet frontend (`http://100.109.192.23:8766/debug`) is now much closer to the
target than the old pre-fix number:

| Path | Synthesis | Audio length | RTF | Clip ratio |
| --- | ---: | ---: | ---: | ---: |
| TTS.cpp Vulkan / `jf_alpha` | `1.09s` to `1.15s` | `5.775s` | `0.189` to `0.199` | `0.0` |

The target remains `0.100 RTF` or lower, so this is not done, but the bottleneck
has moved. The old generate graph had about `17916` nodes and spent almost all
wall time inside Vulkan graph compute:

```text
KOKORO_GRAPH_OPS phase=generate nodes=17916 ADD=6040 MUL_MAT=2719 VIEW=2485 MUL=2029 UNARY:SIGMOID=1854 UNARY:TANH=1237 CONCAT=633 ...
KOKORO_TIMING phase=generate ... compute_submit_ms=3268.58 total_ms=3405.86
```

With `KOKORO_DEBUG_LSTM_STATS=1`, the short benchmark sentence breaks down as:

```text
KOKORO_LSTM_STATS weight=kokoro.duration_predictor.layers.0.lstm.0.weights.0 seq=78 hidden=256 bidirectional=1 node_delta=4076
KOKORO_LSTM_STATS weight=kokoro.duration_predictor.layers.2.lstm.0.weights.0 seq=78 hidden=256 bidirectional=1 node_delta=4081
KOKORO_LSTM_STATS weight=kokoro.duration_predictor.layers.4.lstm.0.weights.0 seq=78 hidden=256 bidirectional=1 node_delta=4081
KOKORO_LSTM_STATS weight=kokoro.duration_predictor.duration_lstm.0.weights.0 seq=78 hidden=256 bidirectional=1 node_delta=4070
KOKORO_LSTM_STATS weight=kokoro.duration_predictor.shared_lstm.0.weights.0 seq=231 hidden=256 bidirectional=1 node_delta=12033
KOKORO_LSTM_STATS weight=kokoro.text_encoder.lstm.0.weights.0 seq=78 hidden=256 bidirectional=1 node_delta=4113
```

After narrowing the stats to only the LSTM body, each 78-step bidirectional
LSTM contributes `4071` nodes:

```text
ADD=1412,MUL_MAT=632,VIEW=624,MUL=468,UNARY:SIGMOID=468,UNARY:TANH=312,CONCAT=155
```

The 231-step shared LSTM contributes `12027` nodes:

```text
ADD=4166,MUL_MAT=1856,VIEW=1848,MUL=1386,UNARY:SIGMOID=1386,UNARY:TANH=924,CONCAT=461
```

The LSTM gate-concat rewrite is now enabled by default. It is controlled by
`KOKORO_FUSE_LSTM_GATES`; set `KOKORO_FUSE_LSTM_GATES=0` to return to the
expanded four-gate graph. On the 82-token Japanese phoneme sample it was
byte-identical to the old default PCM with deterministic UV noise enabled. It
reduced generate nodes from `18592` to `17328` after the IM2COL remap and
reduced generate `compute_submit_ms` from `1082.23` to `1070.14`. This is a
safe small cleanup, not the main speedup.

Increasing `GGML_VK_SUBMIT_COUNT` to `100000` so the backend submits much larger
command batches also did not materially change the result: median `3.367s` /
`0.583 RTF`. The bottleneck is therefore below command-buffer batch frequency;
it is the expanded subgraph itself.

The architecture-level bottleneck is dispatch and intermediate tensor traffic
from decomposing recurrent and generator blocks into thousands of tiny ggml ops.
The next optimization boundary must therefore fuse Kokoro-specific subgraphs,
starting with the recurrent LSTM sections:

- Duration prosody LSTMs over token length.
- Generator shared LSTM over duration frames.
- Text encoder output LSTM.

A graph-level gate concat experiment reduces `MUL_MAT` counts but leaves many
`VIEW`, `CONCAT`, activation, and state-update nodes, so it does not change the
architecture enough. The required direction is a real fused Kokoro LSTM backend
op or Kokoro-specific Vulkan pipeline that keeps the recurrent state on device
and emits the full output sequence without per-gate/per-step ggml dispatches.

The intended graph-optimization route is therefore:

1. Recognize Kokoro LSTM subgraphs at build time instead of relying on a generic
   peephole optimizer after graph creation.
2. Prepack the four input/recurrent gate weights and biases once at model load.
3. Replace the expanded LSTM body with a fused op or Vulkan pipeline that
   computes recurrent gate matmul, sigmoid/tanh, state update, and output write
   inside the same execution boundary.
4. Keep the public graph and CPU path compatible by making the fused path
   opt-in until accuracy and speed gates pass.

An initial opt-in graph rewrite exists behind `KOKORO_FUSE_LSTM_SCAN=1`. It
adds a `KOKORO_LSTM_SCAN` op with a CPU reference implementation and replaces
each bidirectional LSTM body with:

```text
CONCAT=25,ADD=2,KOKORO_LSTM_SCAN=2,MUL_MAT=2
```

That reduces the short-sentence LSTM node contribution from `4071` nodes per
78-step LSTM and `12027` nodes for the 231-step shared LSTM to `31` nodes per
bidirectional LSTM. The rewrite remains opt-in because the Vulkan scan path has
not passed the waveform parity gate.

The Vulkan `KOKORO_LSTM_SCAN` shader is now wired into the backend, but current
profiling shows two constraints. First, it is not waveform-equivalent enough:
on the current Japanese benchmark, `KOKORO_FUSE_LSTM_SCAN=1` produced the same
sample count as default but only about `27.7 dB` SNR against the default PCM,
with max absolute sample difference near `0.0565`. New `kokoro-lstm-scan-test`
coverage shows that CPU expanded vs CPU scan, CPU scan vs Vulkan scan, and
Vulkan expanded vs Vulkan scan pass on realistic single-layer shapes including
`hidden=256, seq=78` and `hidden=256, seq=231`. That narrows the issue to small
per-layer floating-point differences being amplified by the full Kokoro graph,
not an obvious indexing bug in the scan op. Second, the current scan shader is
not the right performance boundary after the IM2COL remap: it computes
recurrent matmul serially inside one workgroup instead of using the optimized
matrix kernels. With fused scan enabled, the full CLI path was slightly slower
than default on the benchmark sample. Therefore this path must stay
experimental until waveform parity and performance are both fixed.

Before the IM2COL remap, `KOKORO_DEBUG_GRAPH_OPS=1` showed the remaining
generate graph was dominated by generator Conv1D lowering:

```text
KOKORO_GRAPH_OPS phase=generate nodes=1880 ADD=466 RESHAPE=236 MUL_MAT=235 TRANSPOSE=182 MUL=175 CONT=143 IM2COL=78 ...
```

A `GGML_VULKAN_PERF=ON` profiling build showed the largest bucket:

```text
IM2COL: 78 x 26.846 ms
CONV_TRANSPOSE_1D: 5 x 70.322 ms
CONT: 143 x 2.657 ms
```

The `CONV_TRANSPOSE_1D` shader was then tightened to avoid scanning every input
time step for each output point. It now computes the exact valid `in_t` range
for the current `out_t` and preserves the original increasing-`in_t`
accumulation order. `vulkan-conv-transpose-1d-test` now covers both toy cases
and the two Kokoro upsample shapes:

```text
Kokoro up0 [114,512] -> [1140,256], k=20, stride=10, padding=5
Kokoro up1 [1140,256] -> [6840,128], k=12, stride=6, padding=3
```

On the 82-token Japanese phoneme sample, this produced byte-identical PCM
against the previous default output with deterministic UV noise enabled, while
default generate `compute_submit_ms` improved from `3634.97` to `3401.61`.
The profiling build now reports:

```text
CONV_TRANSPOSE_1D: 5 x 21.772 ms
IM2COL: 78 x 30.165 ms
```

So ConvTranspose is no longer the first target. The remaining generator hot path
is the Conv1D lowering: materialized `IM2COL`, the following `MUL_MAT`, and the
layout churn around those ops.

The generic Vulkan `IM2COL` shader was then remapped for coalesced writes. The
old dispatch varied output time across neighboring invocations, so writes into
the materialized im2col matrix jumped by `CHW` elements. The new dispatch varies
the `CHW` column inside one im2col row:

```text
x = CHW
y = OW * OH
z = batch
```

This keeps the output layout unchanged and preserves the following optimized
`MUL_MAT` path. New coverage was added through `vulkan-im2col-test`, covering
1D padded/dilated and 2D padded im2col against CPU. The Kokoro Conv1D and full
CLI checks also pass.

On the same 82-token Japanese phoneme sample, PCM stayed byte-identical against
the pre-remap output with deterministic UV noise. The generate phase improved:

```text
Default after ConvTranspose range fix:          compute_submit_ms=3401.61
Default after ConvTranspose + IM2COL remap:     compute_submit_ms=1101.82
Default repeat after IM2COL remap:              compute_submit_ms=1107.56
AdaIN+Snake after ConvTranspose + IM2COL remap: compute_submit_ms=955.618
```

The profiling build now reports:

```text
IM2COL: 78 x 1.012 ms
CONV_TRANSPOSE_1D: 5 x 23.747 ms
```

The remaining generator bottleneck has shifted from im2col materialization to
the following `MUL_MAT` kernels and the many small LSTM/elementwise dispatches.

An additional opt-in rewrite exists behind `KOKORO_FUSE_CONV1D=1`. It adds a
`KOKORO_CONV_1D` op with CPU reference and Vulkan implementations and replaces
Kokoro's `ggml_conv_1d()` calls before they are lowered into
`IM2COL -> MUL_MAT -> RESHAPE`. With both fused flags enabled, the short
benchmark generate graph becomes:

```text
KOKORO_GRAPH_OPS phase=generate nodes=1568 ADD=466 TRANSPOSE=182 MUL=175 MUL_MAT=157 CONT=143 KOKORO_CONV_1D=78 ...
```

Measured through the debug frontend on the Radeon 780M iGPU, this improved
median synthesis from `3.360s / 0.582 RTF` on the default graph to
`3.129s / 0.542 RTF` with fused Conv1D enabled. The waveform sanity values
stayed normal (`peak` about `0.345`, `rms` about `0.065`, no clipping).

The direct Conv1D shader is correctness-first, not the final performance
kernel. Perf profiling shows `KOKORO_CONV_1D: 78 x 27.491 ms`, which is close
to the old `IM2COL` cost by itself. The current speedup comes mostly from
removing the follow-on large `MUL_MAT` and reshape work, not from a faster
convolution kernel. The next graph-optimization step should be a tiled
GEMM-like Conv1D shader or a fused Conv1D+bias/activation family; a naive
one-thread-per-output direct convolution is only a proof of direction.

Accuracy is a hard gate for this work:

- Keep CPU output as the reference for deterministic, seeded runs.
- Use `KOKORO_DEBUG_UV_NOISE_SEED` when comparing CPU and Vulkan so Kokoro's
  random source excitation does not hide backend error.
- Compare intermediate tensor summaries around `f0_out`, `n_out`,
  `kokoro_sing`, `kokoro_har_stft`, `kokoro_out_conv`, and `after_res_gen`.
- Preserve waveform sanity: sample rate `24000`, matching duration, no NaN/Inf,
  no clipping, and stable peak/RMS.
- Do not accept lower precision or approximations unless the seeded CPU/Vulkan
  comparison and listening tests show no regression.

## Upstream ggml Coverage

The following Kokoro-relevant operations are already present in upstream ggml
and were checked with upstream `test-backend-ops`:

| Operation | Upstream API | Vulkan status |
| --- | --- | --- |
| cumulative sum | `ggml_cumsum` | supported |
| 1D transposed convolution | `ggml_conv_transpose_1d` | supported |
| round | `ggml_round` | supported as unary round |
| linear/bilinear upscale | `ggml_upscale` / `ggml_interpolate` | supported through `GGML_OP_UPSCALE` |

Validated commands:

```bash
cmake -S /home/kevinzhow/github/ggml -B /home/kevinzhow/github/ggml/build-vulkan -DGGML_VULKAN=ON -DGGML_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /home/kevinzhow/github/ggml/build-vulkan --target test-backend-ops -j$(nproc)
/home/kevinzhow/github/ggml/build-vulkan/bin/test-backend-ops test -b Vulkan0 -o CUMSUM
/home/kevinzhow/github/ggml/build-vulkan/bin/test-backend-ops test -b Vulkan0 -o CONV_TRANSPOSE_1D
/home/kevinzhow/github/ggml/build-vulkan/bin/test-backend-ops test -b Vulkan0 -o ROUND
/home/kevinzhow/github/ggml/build-vulkan/bin/test-backend-ops test -b Vulkan0 -o UPSCALE
```

## Patched TTS.cpp ggml Coverage

The local patched ggml backend was extended and validated with:

```bash
cmake --build /home/kevinzhow/github/TTS.cpp/build-ggml-vulkan-tests --target test-backend-ops -j$(nproc)
/home/kevinzhow/github/TTS.cpp/build-ggml-vulkan-tests/bin/test-backend-ops test -b Vulkan0 \
  -o ADD,SCALE,MUL,DIV,UNARY,GELU,SILU,TANH,SIN,LEAKY_RELU,REPEAT,CPY,GET_ROWS,ROPE,MUL_MAT,SOFT_MAX,RMS_NORM,NORM,IM2COL,CLAMP,CONCAT,SUM_ROWS,SQR,ROUND,RECIPROCAL,MOD,CUMSUM,UPSCALE,UPSCALE_LINEAR,CONV_TRANSPOSE_1D,STEP,EXP,SIGMOID,STFT,AA_STFT,ISTFT,AA_ISTFT,CONT
```

All filtered tests pass against the CPU backend on the local Vulkan0 device.
The latest local patched ggml test run reports Vulkan0 as an NVIDIA GeForce
RTX 3060 and passes `1929/1929` filtered op tests.

Implementation notes:

- `CUMSUM` needed an explicit 512-wide local size to match the dispatch helper.
- `UPSCALE_LINEAR` exposed and fixed a CPU reference bug where channels were
  skipped when CPU threading was enabled.
- `CONV_TRANSPOSE_1D` needed a dedicated `{time, output-channel, 1}` dispatch
  shape and currently mirrors TTS.cpp's existing CPU semantics, including its
  historical channel-mixing behavior.
- `STEP` and `EXP` use the existing contiguous unary shader ABI.
- `SILU`, `GELU`, `TANH`, `SIGMOID`, and `EXP` are covered through
  `GGML_OP_UNARY`; Kokoro's `LEAKY_RELU` also has a Vulkan pipeline and backend
  op coverage.
- `STFT` / `AA_STFT` / `ISTFT` / `AA_ISTFT` use direct DFT kernels. Kokoro uses
  `n_fft=20` and `hop=5`, so this avoids shape-specialized compilation while
  keeping waveform math in float32.
- Plain complex `ISTFT` now has a Vulkan pipeline and regression test. While
  adding it, the CPU reference was fixed to use scratch buffers rather than
  mutating `src0` in place across overlapping threaded frame ranges.
  Regression coverage includes full complex, one-sided complex, full
  abs/angle, and one-sided abs/angle layouts.
- Kokoro's previous `uv_noise_compute` host callback was replaced with graph
  primitives: `ADD`, `STEP`, `REPEAT`, `MUL`, and scalar views from
  `uv_noise_data`.
- Dia's CFG logit correction no longer uses `ggml_map_custom2`. It is now
  expressed as `cond + scale * (cond + (-1 * uncond))` so the graph only needs
  Vulkan-covered `ADD` and `SCALE` ops. `dia-cfg-scale-test` covers the formula
  and asserts that the graph contains neither custom map ops nor `GGML_OP_SUB`.
- Parler's T5 conditional text encoder now follows the active model backend
  instead of being hard-wired to CPU. Its attention mask and relative position
  bucket inputs are staged through `ggml_backend_tensor_set`, so they work with
  Vulkan buffers instead of relying on host-visible tensor data.
- Parler's cross-attention K/V precompute now uses the model's selected backend
  instead of forcing a CPU-only scheduler. Its computed K/V tensors are copied
  back into the persistent model buffer through backend tensor APIs, so the path
  works whether weights live in CPU or Vulkan buffers. The temporary T5 response
  is now a real stack object rather than an uninitialized pointer.
- Runtime inputs for Kokoro, Dia, Parler, Parler T5, Orpheus, DAC, and SNAC are
  assigned to the active accelerator backend and staged through
  `ggml_backend_tensor_set`. This removes hidden CPU placement caused by
  unassigned graph inputs and avoids direct writes to non-host-visible Vulkan
  tensor buffers.
- I32 token slicing in DAC and SNAC no longer materializes `CONT(view)` copies.
  Vulkan currently has F32/F16 copy pipelines but no I32 copy pipeline, so those
  copies forced CPU placement under the scheduler. The token views are now fed
  directly to `GET_ROWS`.
- CLI `--max-tokens` is now honored by Parler and Orpheus as a temporary
  generation cap, which gives deterministic smoke tests without waiting for
  random EOS.
- TTS.cpp model weight loading now aligns every tensor offset to the selected
  backend buffer alignment. This fixes a Vulkan-only failure where an F32
  Kokoro voice embedding view could land at a non-F32-aligned byte offset after
  F16 tensors, causing garbage values in the duration graph.
- `model-buffer-alignment-test` covers both direct host tensor loading and
  backend-tensor persistence through `set_tensor_from_backend_tensor`, which is
  the helper used by Parler's cross-attention precompute.
- Accelerator contexts skip worst-case graph reservation during post-load. The
  theoretical Kokoro worst case uses `max_context_length *
  max_duration_per_token` and attempted a `5.8GB` Vulkan compute allocation on
  the local RTX 3060; Parler T5 showed the same failure mode after runtime
  inputs were correctly assigned to Vulkan. Actual request graphs are still
  allocated by ggml's scheduler and validated under strict mode.

Validated end-to-end command:

```bash
TTS_BACKEND=vulkan TTS_BACKEND_STRICT=1 \
  /home/kevinzhow/github/TTS.cpp/build-vulkan/bin/tts-cli \
  -mp /home/kevinzhow/github/TTS.cpp/build-vulkan/models/Kokoro_no_espeak_Q4.gguf \
  -p 'Hello world.' \
  -v af_heart \
  -sp /tmp/kokoro-vulkan-final.wav
```

The run produced `38400` samples at `24000 Hz`, matching the CPU path's
`1.6s` output length. Before the alignment fix the same prompt saturated
duration prediction and produced `450000` samples / `18.75s`.

Validated service-style repeated requests:

```bash
TTS_BACKEND_STRICT=1 \
  /home/kevinzhow/github/TTS.cpp/build-vulkan/bin/tts-server \
  --backend vulkan \
  --model-path /home/kevinzhow/github/TTS.cpp/build-vulkan/models/Kokoro_no_espeak_Q4.gguf \
  --voice af_heart \
  --host 127.0.0.1 \
  --port 18082 \
  --n-parallelism 1 \
  --n-http-threads 2
```

Four same-process requests of different lengths returned HTTP 200 with these
WAV durations:

| Request | Frames | Seconds | curl total |
| --- | ---: | ---: | ---: |
| `Hello world.` | 38400 | 1.600 | 0.321633s |
| medium prompt | 77400 | 3.225 | 0.282319s |
| longer prompt | 275400 | 11.475 | 1.297317s |
| `Hello world.` again | 38400 | 1.600 | 0.126640s |

The server log showed one shader compilation during startup and only request
logs afterwards, with no per-length Vulkan shader recompilation and no
worst-case reserve OOM.

Additional same-process voice coverage was validated through `/v1/audio/speech`
using `af_heart`, `am_adam`, `bf_emma`, and `bm_george`; all returned HTTP 200
with non-empty 24 kHz mono WAV output. A long chunking request of nine repeated
sentences returned HTTP 200 with `2241000` frames / `93.375s` of audio in
`10.255639s`, with no extra shader compilation or worst-case reserve OOM.

Additional real-model strict Vulkan smoke tests:

| Model | Command shape | Output | Total time |
| --- | --- | --- | ---: |
| Dia `Dia_Q4_DAC_F16.gguf` | `--prompt '[S1] Hello.' --topk 35 --temperature 1.3 --max-tokens 32` | `7680` frames / `0.17415s` at 44.1 kHz | `3034.33 ms` |
| Parler `Parler_TTS_mini_Q5.gguf` | `--prompt 'Hello world.' --max-tokens 32` | `10240` frames / `0.2322s` at 44.1 kHz | `865.60 ms` |

## Remaining Backend Gaps

The remaining unvalidated area is not an uncovered Vulkan operator in the
current smoke set; it is model availability versus local device capacity:

| TTS.cpp usage | Current source | Needed direction |
| --- | --- | --- |
| Orpheus real synthesis | model size / local GPU memory | published `Orpheus.gguf` is `15191828320` bytes, larger than the local 12 GB Vulkan device memory; validate when a quantized GGUF exists or on a larger GPU |

## Design Direction

The preferred path is to move TTS.cpp toward current upstream ggml rather than
backporting modern Vulkan kernels into the old patched submodule. Current ggml
already includes several TTS-critical Vulkan kernels and a more complete backend
test suite.

For Kokoro specifically, the graph should be split and stabilized:

- Keep model weights in backend buffers selected by ggml.
- Use Vulkan kernels for encoder, duration, pitch, convolution, round, cumsum,
  reciprocal, mod, upscale, STFT, and iSTFT-style waveform reconstruction where
  coverage already exists.
- Keep host callbacks out of the Kokoro graph before expecting full GPU
  execution.
- Keep waveform reconstruction numerically conservative: MLX-style Kokoro
  implementations keep the network in lower precision while retaining float32
  for waveform reconstruction, STFT/iSTFT-style math, and overlap-add.
- Prefer sentence or bounded-chunk execution for long text so graph shapes stay
  within a small set of reusable plans.

This is still experimental until it is validated across more local GGUF model
families. The current work establishes backend selection, removes Kokoro's host
uv/noise callback and Dia's CFG host callback, moves Parler's T5 conditional
encoder and cross-attention precompute onto the selected backend, validates the
patched ggml Vulkan op set against the CPU backend, checks strict graph
placement to prevent silent CPU fallback, and proves real Kokoro, Dia, and
Parler GGUF synthesis through strict Vulkan.
