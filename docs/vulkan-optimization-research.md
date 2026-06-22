# Kokoro Vulkan Optimization Research

Date: 2026-06-22

Scope: Kokoro Japanese synthesis in TTS.cpp on the local AMD Radeon 780M
Vulkan/RADV iGPU path. This document is a research-backed execution plan; raw
measurements and operator coverage status remain in `docs/vulkan-backend-status.md`.

## Current Local Evidence

The current short Japanese benchmark sentence is:

```text
赤い血は白い雪のうえでとてもきれいに見えたので女王さまは思いました。
```

Earlier local measurements in `vulkan-backend-status.md` showed the pre-fix
debug-frontend baseline:

| Backend | Median synthesis | Audio length | RTF | Clip ratio |
| --- | ---: | ---: | ---: | ---: |
| TTS.cpp CPU | `2.252s` | `5.775s` | `0.390` | `0.0` |
| TTS.cpp Vulkan / Radeon 780M | `3.360s` | `5.775s` | `0.582` | `0.0` |

After the ConvTranspose range fix, IM2COL remap, and default LSTM gate fusion,
the current source-run CLI measurement on an 82-token Japanese phoneme sample is:

| Backend / mode | Duration compute | Generate compute | Total CLI time | Output length |
| --- | ---: | ---: | ---: | ---: |
| CPU | `66.600 ms` | `3212.53 ms` | `4614.65 ms` | `6.000s` |
| Vulkan default | `101.756 ms` | `1070.14 ms` | `~2.1s` | `6.000s` |
| Vulkan + `KOKORO_FUSE_ADAIN_SNAKE=1` | `101.910 ms` | `940.287 ms` | `~2.0s` | `6.000s` |

The important finding is no longer simply that Vulkan is slower than CPU. The
generator memory-layout bottleneck was real and has been mostly removed; the
remaining work is now in matrix kernels and many small recurrent/elementwise
dispatches.

Observed graph/profiling evidence:

```text
KOKORO_GRAPH_OPS phase=generate nodes=17916 ADD=6040 MUL_MAT=2719 VIEW=2485 MUL=2029 UNARY:SIGMOID=1854 UNARY:TANH=1237 CONCAT=633 ...
KOKORO_TIMING phase=generate ... compute_submit_ms=3268.58 total_ms=3405.86
```

The experimental LSTM scan rewrite proves node-count reduction potential, but
it is not currently accuracy-safe on Vulkan: an 82-token Japanese phoneme sample
measured only about `28.4 dB` SNR against the default PCM when
`KOKORO_FUSE_LSTM_SCAN=1` was enabled. Even with that unsafe path, the generate
phase is not materially faster after the IM2COL remap. By contrast,
`KOKORO_FUSE_LSTM_GATES` is now enabled by default because the gate-concat
rewrite is byte-identical to the previous default output on the same sample and
reduces generate nodes from `18592` to `17328`:

```text
KOKORO_GRAPH_OPS phase=generate nodes=17328 VIEW=5165 ADD=4330 MUL=2107 ... MUL_MAT=879 ...
```

The largest profiled generate buckets were:

```text
IM2COL: 78 x 26.846 ms
CONV_TRANSPOSE_1D: 5 x 70.322 ms
CONT: 143 x 2.657 ms
```

The opt-in direct Conv1D path (`KOKORO_FUSE_CONV1D=1`) reduced graph node count,
but the latest focused benchmark shows it is not a valid performance path. The
shader is a correctness probe, not the kernel we should optimize around:

```text
default generate: nodes=5228 IM2COL=78 compute_submit_ms=525.590
fused Conv1D:     nodes=4916 KOKORO_CONV_1D=78 compute_submit_ms=1028.380
```

The matching microbenchmark uses real generator hotspot shapes from the layout
trace and compares CPU, Vulkan `im2col + matmul`, and Vulkan direct fused
Conv1D. The key results on the Radeon 780M were:

| Case | Trace count | Vulkan im2col | Vulkan direct fused |
| --- | ---: | ---: | ---: |
| `res0-k3-d1` `[1140,256]` | 6 | `1.685 ms` | `9.509 ms` |
| `res0-k7-d3` `[1140,256]` | 12 | `3.657 ms` | `15.620 ms` |
| `res0-k11-d5` `[1140,256]` | 6 | `5.715 ms` | `18.579 ms` |
| `res1-k3-d1` `[6841,128]` | 6 | `4.015 ms` | `12.687 ms` |
| `res1-k7-d3` `[6841,128]` | 6 | `7.595 ms` | `18.723 ms` |
| `res1-k11-d5` `[6841,128]` | 12 | `13.864 ms` | `23.015 ms` |

The direct fused results were numerically close to the CPU reference at the
single-op level, but they were slower because one shader invocation computes one
output element and loops over all input channels and kernel taps. The current
Vulkan `im2col + matmul` path pays extra memory traffic, but it still uses the
backend's optimized matrix kernels. That is the performance baseline any new
kernel must beat.

A separate `GGML_VULKAN_PERF=ON` diagnostic build with
`GGML_VK_SUBMIT_COUNT=1` gives a more useful hotspot ranking for the current
default path. Absolute times are inflated because every op is fenced, but the
relative order is still useful:

```text
generate diagnostic:
  IM2COL:             78 x 2.850 ms = 222.300 ms
  ADD:              1648 x 0.077 ms = 126.896 ms
  CONV_TRANSPOSE_1D:   5 x 22.835 ms = 114.175 ms
  MUL:               565 x 0.088 ms = 49.720 ms
  MUL_MAT_VEC:       520 x 0.066 ms = 34.320 ms
  CONT:              143 x 0.192 ms = 27.456 ms
```

This points to two complementary optimization layers: reduce materialized
`IM2COL` traffic with a tiled or implicit-GEMM Conv1D kernel, and fuse the
surrounding elementwise/layout work so we do not spend hundreds of dispatches on
small `ADD`, `MUL`, and `CONT` operations.

The later Snake1D transpose fusion experiment is a useful negative result:
reducing node count alone is not enough. Fusing `transpose + cont + snake` into
one op reduced graph nodes, but slowed end-to-end synthesis because the fused
kernel read a transposed channel-major view with poor coalescing. The next step
must own the tensor layout across an entire generator block, not fuse isolated
ops across an unfavorable layout boundary.

An AdaIN+Snake fusion was added as an opt-in probe behind
`KOKORO_FUSE_ADAIN_SNAKE=1`. It fuses:

```text
x + x * gamma + beta
snake_1d(alpha, cont(transpose(x)))
```

into one CPU/Vulkan op, with unit coverage for CPU composition parity and
Vulkan parity. The real Kokoro CLI measurements are useful but modest:

| Input | Mode | Nodes | Vulkan `compute_submit_ms` | PCM diff vs default |
| --- | --- | ---: | ---: | ---: |
| `akaika` | default | 5228 | `420.556` | baseline |
| `akaika` | `KOKORO_FUSE_SNAKE=1` | 4892 | `399.297` | SNR `68.2 dB` |
| `akaika` | `KOKORO_FUSE_ADAIN_SNAKE=1` | 4748 | `387.811` | SNR `68.0 dB` |
| 82-token Japanese phoneme sample | default | 18592 | `3634.97` | baseline |
| 82-token Japanese phoneme sample | `KOKORO_FUSE_SNAKE=1` | 18256 | `3481.03` | SNR `68.3 dB` |
| 82-token Japanese phoneme sample | `KOKORO_FUSE_ADAIN_SNAKE=1` | 18112 | `3482.34` | SNR `68.2 dB` |

The short input sees about `7.8%` compute-submit improvement for AdaIN+Snake.
The longer representative sample sees about `4.2%`, almost identical to
Snake-only fusion. This means the fused op is safe and useful as a controlled
experiment, but it is not the main path to CPU parity. It removes dispatches
around Snake, while the heavy time remains in convolution/im2col, transposed
convolution, large matmul regions, and thousands of small recurrent/elementwise
dispatches.

A separate `GGML_VULKAN_PERF=ON` build confirms this shape of the problem on
the 82-token sample. Absolute values are inflated by profiling fences, but the
hot buckets are still directional:

```text
IM2COL:                  78 x 30.094 ms
CONV_TRANSPOSE_1D:        5 x 83.800 ms
KOKORO_ADAIN_SNAKE_1D_T: 48 x  7.674 ms
ADD:                   6130 x  0.061 ms
MUL:                   1963 x  0.057 ms
SIGMOID:               1932 x  0.056 ms
TANH:                  1289 x  0.056 ms
```

The actionable conclusion is that isolated elementwise fusion is only a first
layer. The larger win needs a generator-block-level layout strategy: keep data
in the layout consumed by the next Conv1D, avoid materialized `IM2COL` where a
tiled implicit-GEMM kernel can win, and avoid fusing across a boundary that
forces uncoalesced reads.

The ConvTranspose1D Vulkan shader now has a more meaningful kernel-level fix.
The previous shader computed each `(out_t, out_channel)` by scanning every input
time step and discarding almost all of them with a bounds check. For stride-10
and stride-6 Kokoro upsample layers, only a small contiguous input-time range
can contribute to a given output time. The shader now computes that valid range
directly and keeps the original increasing-`in_t` accumulation order.

Correctness gates:

```text
vulkan-conv-transpose-1d-test
  dense toy case
  depthwise toy case
  Kokoro up0 [114,512] -> [1140,256], k=20, stride=10, padding=5
  Kokoro up1 [1140,256] -> [6840,128], k=12, stride=6, padding=3
  AA_STFT / AA_ISTFT parity
```

The 82-token Japanese phoneme sample produced byte-identical PCM against the
pre-change default output when `KOKORO_DEBUG_UV_NOISE_SEED=1` was set. Timing
improved without changing graph shape:

| Mode | Generate nodes | Vulkan `compute_submit_ms` |
| --- | ---: | ---: |
| Default before ConvTranspose range fix | 18592 | `3634.97` |
| Default after ConvTranspose range fix | 18592 | `3401.61` |
| `KOKORO_FUSE_ADAIN_SNAKE=1` before range fix | 18112 | `3482.34` |
| `KOKORO_FUSE_ADAIN_SNAKE=1` after range fix | 18112 | `3160.57` |

The profiling build also shows the targeted bucket falling from the earlier
`83.8 ms/op` measurement to `21.772 ms/op`:

```text
CONV_TRANSPOSE_1D: 5 x 21.772 ms
IM2COL:           78 x 30.165 ms
```

This is the right kind of optimization: it removes wasted work inside a hot
kernel while preserving the graph and output. The remaining dominant bucket is
now even more clearly `IM2COL + MUL_MAT` for generator Conv1D.

The next fix was even more important: remap the generic Vulkan `IM2COL` shader
so adjacent threads write adjacent output addresses. The old mapping varied
`ix` across neighboring invocations, which writes with a stride of `CHW` into
the materialized im2col matrix. The new mapping dispatches:

```text
x = CHW
y = OW * OH
z = batch
```

and computes `(ic, ky, kx, oh, ix)` from that. The output tensor layout is
unchanged, but each workgroup writes a contiguous slice of one im2col row. This
keeps the optimized `MUL_MAT` path and only removes bad memory coalescing in the
materialization step.

Correctness gates:

```text
vulkan-im2col-test
  1D padded+dilated im2col vs CPU
  2D padded im2col vs CPU
kokoro-conv-1d-test
  Kokoro Conv1D default im2col path vs CPU/fused references
82-token Japanese phoneme CLI sample
  byte-identical PCM vs pre-remap output with deterministic UV noise
```

The same 82-token sample shows the payoff:

| Mode | Generate nodes | Vulkan `compute_submit_ms` |
| --- | ---: | ---: |
| Default before ConvTranspose range fix | 18592 | `3634.97` |
| Default after ConvTranspose range fix | 18592 | `3401.61` |
| Default after ConvTranspose + IM2COL remap | 18592 | `1101.82`, repeat `1107.56` |
| `KOKORO_FUSE_ADAIN_SNAKE=1` after ConvTranspose + IM2COL remap | 18112 | `955.618` |

The profiling build confirms the targeted bucket collapsed:

```text
before remap:
  IM2COL: 78 x 30.165 ms

after remap:
  IM2COL: 78 x 1.012 ms
  CONV_TRANSPOSE_1D: 5 x 23.747 ms
```

This changes the remaining optimization problem. The biggest generator bottleneck
is no longer materializing im2col; it is the subsequent matrix products and the
large number of small recurrent/elementwise dispatches.

## Source-Level Layout Map

The local Kokoro source confirms the layout churn is structural, not an
artifact of profiling.

In `src/models/kokoro/model.cpp`, `build_kokoro_generator_res_block()` repeatedly
does:

```text
norm(inpl)
transpose
cont
AdaIN add/mul
Snake
Conv1D
...
norm(cur)
transpose
cont
AdaIN add/mul
Snake
Conv1D
residual add
```

The wrapper `kokoro_snake_1d_t()` also defaults to:

```text
snake_1d(alpha, cont(transpose(input_channels_time)))
```

unless `KOKORO_FUSE_SNAKE=1` is set. That explains why isolated Snake fusion was
a risky boundary: the unfused path materializes a different contiguous layout
before Snake, while the fused path reads the transposed view directly.

`build_generator()` adds more layout transitions around upsample and post
processing:

```text
conv_transpose_1d(weight, cont(transpose(cur)))
...
cur = cont(transpose(div(cur, n_kernels)))
...
out_conv(weight, cont(transpose(cur)))
...
istft(cont(transpose(cur)))
```

The current Vulkan `kokoro_conv_1d.comp` shader maps one invocation to one output
element and loops over all input channels and kernel taps. It now supports
strided input, but that also means it can legally read poor layouts. Its memory
access pattern is therefore correctness-first rather than optimized:

```text
input_offset = iw * input_nb0 + ic * input_nb1 + n * input_nb2
dst[row + oc * rows] = sum
```

This is why a tiled Conv1D microkernel must be designed together with a chosen
generator layout and weight prepacking. Merely replacing `IM2COL` with a scalar
direct shader loses to the existing matmul path and cannot reach the target RTF.

An opt-in trace now exists to make this visible in real graph builds:

```bash
KOKORO_DEBUG_GENERATOR_LAYOUT=1
```

It prints `KOKORO_GENERATOR_LAYOUT` lines containing label, tensor name, op,
shape, byte stride, element size, contiguous flag, transposed flag, and view
flag. This is intentionally graph-metadata only; it does not change synthesis
behavior.

## External Findings

### Vulkan runtime overhead is real but secondary here

Khronos' Vulkan pipeline barrier sample shows why over-conservative barriers can
serialize work and create pipeline bubbles. It recommends keeping source stages
as early as possible, destination stages as late as possible, avoiding full
pipeline-draining patterns, and only using intra-queue barriers when needed:

https://docs.vulkan.org/samples/latest/samples/performance/pipeline_barriers/README.html

AMD's RDNA performance guide gives matching advice: minimize GPU submissions,
avoid just-in-time PSO compilation, reuse PSO caches, minimize barriers, batch
barrier groups, use large memory allocations with suballocation, and minimize
descriptor updates:

https://gpuopen.com/learn/rdna-performance-guide/

For TTS.cpp this does matter, but it is unlikely to produce a 4x-6x speedup by
itself. We already tested larger Vulkan command batch size via
`GGML_VK_SUBMIT_COUNT`; it did not materially change the short benchmark. The
dominant local evidence is inside the generate graph kernels and layout churn.

### Pipeline cache should still be cleaned up

Khronos' pipeline cache sample states that pipeline creation compiles shader
modules internally and can cause runtime frame-time spikes; persistent
`VkPipelineCache` data lets subsequent runs reuse baked state:

https://docs.vulkan.org/samples/latest/samples/performance/pipeline_cache/README.html

For TTS.cpp, pipeline cache is primarily a cold-start and deployment hygiene
item. It should not be confused with the OpenVINO dynamic-shape problem we saw
earlier. ggml's Vulkan shader set is finite and shape-independent unless we add
shape-specialized pipelines; request length changes should not force new Vulkan
pipeline compilation in this backend.

If we add shape-specialized kernels, the cache policy must be explicit:

- Compile only the small, known Kokoro hot-shape set.
- Create pipelines at model load or first warmup, not inside the request path.
- Persist pipeline cache data across process restarts.
- Report pipeline cache hits/misses in debug output.

### Descriptor and buffer management can hurt many-dispatch graphs

Khronos' descriptor management sample calls out that recreating descriptor pools
and sets frequently can be inefficient, and that descriptor management is tied
to how data is packed into `VkBuffer` objects:

https://docs.vulkan.org/samples/latest/samples/performance/descriptor_management/README.html

IREE's Vulkan HAL has a similar direction: descriptor-set dispatch is the
portable baseline, while Buffer Device Address (BDA) dispatch is an optional
faster ABI when supported. IREE also opportunistically uses push descriptors to
reduce descriptor allocation/update overhead and calibrated timestamps for
device-side profiling:

https://iree.dev/guides/deployment-configurations/gpu-vulkan/

For TTS.cpp, descriptor/BDA work is a good second-order track after generator
layout work. With 1500-1800 generate nodes, descriptor churn can matter, but the
largest measured buckets are still Conv1D, ConvTranspose1D, and contiguous
materialization.

### Compiler frameworks optimize at graph/dispatch boundaries, not one op at a time

IREE's dispatch creation passes include elementwise fusion, fusion of multi-use
elementwise producers, split-reduction dispatch formation, and horizontal
contraction fusion:

https://iree.dev/reference/mlir-passes/DispatchCreation/

Apache TVM's optimization tutorials show the same pattern: high-level graph
optimizations and low-level tensor schedule tuning are separate layers. TVM can
selectively fuse `Linear + ReLU`, dispatch certain computation to libraries, and
use MetaSchedule to search tiling/vectorization/thread-binding schedules on real
hardware:

https://tvm.apache.org/docs/how_to/tutorials/customize_opt.html
https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html

The lesson for TTS.cpp is direct: our existing isolated op fusions are useful
probes, but the real backend should form Kokoro-specific dispatch regions:
generator residual blocks, Conv1D + bias + AdaIN + Snake, and generator
upsampling/post blocks.

### Kokoro Core ML confirms the pipeline-surgery direction

The Kokoro Core ML conversion splits the model into stage-specific packages and
fixed-duration buckets. Its model card reports 30 seconds of speech in 379 ms on
an M2 Studio and explains that fixed-shape convolution-heavy stages run on the
accelerator, while branching/dynamic stages stay outside that path. It also
replaces linear layers with `Conv1d(kernel_size=1)` to keep the graph on the
accelerator path:

https://huggingface.co/mattmireles/kokoro-coreml

We cannot copy the hardware policy directly because our stated target is pure
iGPU Vulkan, not CPU/ANE staging. But the architectural lesson applies:

- Static buckets beat open-ended dynamic shapes.
- Accelerator-friendly subgraphs beat monolithic generic graphs.
- Generator math should be represented as convolution/elementwise blocks, not
  thousands of generic scalar ggml nodes.
- Data-dependent control flow should be isolated, or represented as a fused
  domain op with one backend boundary.

## Primary Optimization Thesis

The next real speedup must come from owning the Kokoro generator layout and
forming larger generator dispatch regions.

The current graph alternates between layouts and pays for `TRANSPOSE`, `CONT`,
`IM2COL`, and scalar elementwise ops around Conv1D. A fused op that crosses a
bad layout boundary can make this worse, as the Snake1D experiment showed.

The correct boundary is therefore not:

```text
transpose + cont + snake
```

The correct boundary is closer to:

```text
generator block:
  input layout
  Conv1D / ConvTranspose1D
  bias / AdaIN scale-shift
  Snake
  residual add
  output layout
```

Within that boundary we can choose one layout, prepack weights once, and make
all memory access coalesced.

## Recommended Implementation Plan

### Phase 0: lock down profiling and correctness gates

Before deeper kernel work, keep these gates visible in every benchmark:

- `KOKORO_DEBUG_TIMINGS=1`
- `KOKORO_DEBUG_GRAPH_OPS=1`
- `KOKORO_DEBUG_GRAPH_SHAPES=1`
- `KOKORO_DEBUG_GENERATOR_LAYOUT=1`
- `KOKORO_DEBUG_UV_NOISE_SEED=<fixed>`
- waveform duration, peak, RMS, clip ratio
- backend name and Vulkan device name
- generate-phase op histogram and hot-shape histogram

Add one more profiling layer before major rewrites:

- per-op Vulkan timestamp buckets for `KOKORO_CONV_1D`,
  `CONV_TRANSPOSE_1D`, `CONT`, `TRANSPOSE`, `MUL_MAT`, and future generator
  block ops;
- optional RADV SQTT/RGP capture instructions for pure compute workloads;
- optional shader stats dump path for register pressure/occupancy checks.

Rationale: AMD's occupancy material and RGP documentation emphasize identifying
whether a workload is limited by occupancy, memory latency, or synchronization
before tuning shader structure:

https://gpuopen.com/learn/occupancy-explained/
https://gpuopen.com/rgp/

### Phase 1: choose and enforce a generator layout

Define a generator layout policy explicitly. The likely choices are:

- time-major contiguous: good for output samples and per-time dispatch;
- channel-major contiguous: good for per-channel parameters and existing Conv1D
  weight shape;
- block/tile-major: best long-term for convolution kernels but more invasive.

Do not add more isolated fusions until this is decided. The implementation needs
to prove:

- how each generator tensor is laid out before/after every residual block;
- where `CONT` nodes are introduced;
- where `TRANSPOSE` views become real copies;
- whether Conv1D and Snake read contiguous or strided memory.

Acceptance criterion: a debug run should show fewer layout-materialization nodes
without slowing the fused path. If node count drops but wall time rises, reject
the layout.

### Phase 2: replace scalar Conv1D with shape-specialized tiled kernels

The direct Conv1D shader is correctness-first and one-thread-per-output. It
should not be enabled by default. It should be replaced by a small set of Kokoro
hot-shape kernels, and those kernels must beat the current Vulkan
`im2col + matmul` baseline, not only the direct shader. The hot shapes observed
so far are concentrated enough to justify specialized kernels.

Kernel direction:

- Prepack weights once at model load into the chosen layout.
- Use workgroups that compute a tile of output time x output channels.
- Reuse input window data across output channels.
- Keep inner-channel reductions coalesced.
- Use specialization constants only for stable kernel variants, not for every
  request length.
- Benchmark f32 first; only introduce f16/packed math after seeded CPU/Vulkan
  parity and listening tests pass.

Do not use a generic shader with many uniform-controlled branches if the hot
shape set is small. Khronos' specialization-constants sample shows the compiler
can optimize more aggressively when control-flow constants are known at pipeline
creation time:

https://docs.vulkan.org/samples/latest/samples/performance/specialization_constants/README.html

### Phase 3: fuse generator elementwise around Conv1D

Once Conv1D layout is stable, fuse the surrounding generator math:

- Conv1D output write
- bias
- AdaIN `x + x * gamma + beta`
- Snake activation
- residual add when shape-compatible

This should remove many `ADD`, `MUL`, `UNARY`, `TRANSPOSE`, and `CONT` nodes.
The success metric is not only fewer nodes; it must reduce generate compute
time and preserve waveform sanity.

### Phase 4: optimize ConvTranspose1D and post-generator path

`CONV_TRANSPOSE_1D` appears only a few times but has a large per-op cost.
After Conv1D blocks are improved, revisit:

- tiled ConvTranspose1D;
- fused upsample + post-conv where applicable;
- harmonic source / STFT / iSTFT memory traffic;
- whether any post-generator op forces a layout bounce.

### Phase 5: fix or retire LSTM scan after generator work

The LSTM gate-concat rewrite is enabled by default and can be disabled with
`KOKORO_FUSE_LSTM_GATES=0`. It is byte-identical to the previous default output
on the 82-token Japanese phoneme sample and gives a small node-count/timing
cleanup.

The deeper LSTM scan rewrite collapses thousands of recurrent subgraph nodes
into one backend op, but it is not currently accuracy-safe on Vulkan. Local
measurements with `KOKORO_FUSE_LSTM_SCAN=1` produced roughly `28.4 dB` PCM SNR
against the default path on an 82-token Japanese phoneme sample, so it must not
be enabled as an optimization until parity is fixed. It also did not beat the
default gate-fused graph after the IM2COL remap, because its shader computes the
recurrent matmul serially inside one workgroup instead of using the optimized
matrix kernels.

Keep `KOKORO_FUSE_LSTM_SCAN=1` as an opt-in diagnostic path only. Revisit it
after generator work. Potential later work:

- fix CPU/Vulkan scan parity against the expanded ggml LSTM body;
- improve recurrent matmul tiling;
- pack input/recurrent gate weights for Vulkan;
- use one dispatch per direction/layer;
- keep recurrent state in registers/shared memory where possible.

### Phase 6: runtime hygiene after graph/kernel wins

After generator block kernels are in place, revisit runtime overhead:

- persistent `VkPipelineCache` serialized to disk;
- pre-create all known Kokoro pipelines at model load or warmup;
- descriptor set caching/pooling for repeated graph shapes;
- evaluate BDA-style buffer access if the local RADV device exposes it;
- avoid needless intra-queue barriers;
- minimize command submissions without hiding per-op hot spots.

Benchmark-only driver toggles to try on RADV:

- `RADV_PERFTEST=cswave32`
- `RADV_PERFTEST=nogttspill`
- `RADV_PROFILE_PSTATE=peak`

Mesa documents `cswave32` as enabling wave32 for compute shaders and
`nogttspill` as disabling GTT spilling. These should be treated as diagnostic
toggles, not product requirements:

https://docs.mesa3d.org/envvars.html

## Things Not To Do Next

- Do not spend more time adding isolated unary fusions before the generator
  layout is settled.
- Do not accept a fused op only because graph node count dropped.
- Do not globally switch to f16 or approximate math without seeded CPU/Vulkan
  parity and listening tests.
- Do not solve a compute-kernel bottleneck by adding an unbounded cache keyed by
  input length.
- Do not route hot generator blocks through CPU if the target path is pure iGPU
  Vulkan.

## Near-Term Experiments

1. Add a layout trace for generator tensors.
   - Output: tensor name/op, shape, stride, contiguous flag, and whether a
     materializing `CONT` follows.
   - Decision: choose the least-copy layout for block fusion.

2. Build two Conv1D microkernels for the hottest shape.
   - Variant A: time-major output tile.
   - Variant B: channel-major output tile.
   - Compare against both current Vulkan `im2col + matmul` and current
     `KOKORO_CONV_1D` with fixed input/output buffers.

3. Implement one fused generator block behind a new opt-in flag.
   - Keep CPU reference graph unchanged.
   - Compare against the default Vulkan graph and the direct fused Conv1D probe.
   - Reject it if node count drops but generate `compute_submit_ms` rises.

4. Add persistent Vulkan pipeline cache instrumentation.
   - Report whether pipeline cache is loaded, saved, and hit.
   - Confirm request length changes do not cause runtime pipeline creation.

5. Run RADV diagnostics only after the above measurements.
   - Compare default vs `cswave32` vs `nogttspill`.
   - Record whether changes affect all kernels or only specific fused kernels.

## Success Criteria

Short-term success:

- Vulkan fused path beats current CPU on the Japanese benchmark.
- No waveform regression: matching duration, no NaN/Inf, no clipping, stable
  peak/RMS, and acceptable listening test.
- Repeated runs with varied text length avoid per-request pipeline compilation.

Medium-term success:

- Generate phase falls well below `1s` for the 5.775s benchmark audio.
- Vulkan RTF is below `0.20` on Radeon 780M.
- The backend remains pure iGPU Vulkan for the hot synthesis path.

Long-term target:

- Reach roughly `0.10` RTF or better on the same local iGPU path without
  sacrificing Japanese pronunciation/prosody quality.
