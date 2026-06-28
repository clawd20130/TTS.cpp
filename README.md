## TTS.cpp

[Roadmap](https://github.com/users/mmwillet/projects/1) / [TTS GGML fork](https://github.com/clawd20130/ggml/tree/tts)

### Purpose and Goals

The general purpose of this repository is to support real time generation with
open source TTS (_text to speech_) models across common device architectures
using the [GGML tensor library](https://github.com/ggerganov/ggml). Rapid STT
(_speech to text_), embedding generation, and LLM generation are well supported
on GGML via [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and
[llama.cpp](https://github.com/ggerganov/llama.cpp). This repository provides a
similarly portable runtime for TTS models.

This fork includes native GGML runtime pieces for Style-Bert-VITS2 pipelines.
The current focus is accurate local Style-Bert-VITS2 inference with CPU, Metal,
and Vulkan backends while keeping text frontend logic outside of this C++
runtime.

### Supported Functionality

**Warning:** TTS.cpp is still evolving. The generic text-to-speech CLI path and
the Style-Bert-VITS2 sidecar path have different input contracts; see the
Style-Bert section below before wiring an application to it.

#### Fork Highlights

- Runtime backend selection through `--backend auto|cpu|metal|vulkan`,
  `TTS_BACKEND`, `TTS_DEVICE`, and `TTS_BACKEND_STRICT`.
- Style-Bert-VITS2 GGUF runtime for decoder, latent, and front-end graph stages.
- Style-Bert-VITS2 JP-BERT GGUF feature extractor.
- HTTP server endpoints for Style-Bert tensor/symbol synthesis and JP-BERT
  feature extraction.
- Vulkan backend integration through our GGML fork, including conservative
  accuracy defaults for Style-Bert-VITS2.
- Metal backend support for local macOS runs, including a tiled Style-Bert
  conv1d path for large outputs.
- GGUF conversion scripts for Style-Bert-VITS2 voice models and JP-BERT.

#### Model Support

Style-Bert-VITS2 is the primary model for the newer GGML backend work. The older
models remain available through the generic CLI/server text path.

| Model | CPU | Metal | Vulkan | Quantization | GGUF files |
| --- | --- | --- | --- | --- | --- |
| [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) | &check; | &check; | &check; | &cross; | [kevinzhow/style-bert-vits2-gguf](https://huggingface.co/kevinzhow/style-bert-vits2-gguf) |
| [Parler TTS Mini](https://huggingface.co/parler-tts/parler-tts-mini-v1) | &check; | &check; | experimental | &check; | [here](https://huggingface.co/mmwillet2/Parler_TTS_GGUF) |
| [Parler TTS Large](https://huggingface.co/parler-tts/parler-tts-large-v1) | &check; | &check; | experimental | &check; | [here](https://huggingface.co/mmwillet2/Parler_TTS_GGUF) |
| [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) | &check; | experimental | experimental | &check; | [here](https://huggingface.co/mmwillet2/Kokoro_GGUF) |
| [Dia](https://github.com/nari-labs/dia) | &check; | &check; | &cross; | &check; | [here](https://huggingface.co/mmwillet2/Dia_GGUF) |
| [Orpheus](https://github.com/canopyai/Orpheus-TTS) | &check; | &cross; | &cross; | &cross; | [here](https://huggingface.co/mmwillet2/Orpheus_GGUF) |

Additional Model support will initially be added based on open source model performance in both the [old TTS model arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) and [new TTS model arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2) as well as the availability of said models' architectures and checkpoints.

#### Functionality

| Functionality | Status |
| --- | --- |
| Basic CPU generation | Supported |
| Metal acceleration | Supported through GGML Metal, primarily for macOS |
| Vulkan acceleration | Supported when built with `GGML_VULKAN=ON` |
| Runtime backend selection | `--backend`, `TTS_BACKEND`, `TTS_DEVICE`, `TTS_BACKEND_STRICT` |
| Server support | `tts-server`, including generic OpenAI-like speech and Style-Bert-specific endpoints |
| Style-Bert-VITS2 text frontend | Out of scope for TTS.cpp; callers provide normalized phones, tones, language IDs, and BERT features |
| Streaming audio | Not supported |
| CUDA support | Not currently wired as a first-class TTS.cpp target |

### Installation

#### Requirements:

* Local GGUF model files. See [py-gguf](./py-gguf/README.md) for conversion
  details.
* CMake (>= 3.14).
* C++20-capable compiler for the library. The example server currently requests
  C++17 features.
* GGML from this repository's `ggml` submodule, or an external GGML tree passed
  with `-DTTS_GGML_SOURCE_DIR=/path/to/ggml`.
* Vulkan SDK when building with `GGML_VULKAN=ON`.
* Xcode Command Line Tools for macOS/Metal builds.

#### GGML Branch

The active GGML customizations live in
[`https://github.com/clawd20130/ggml`](https://github.com/clawd20130/ggml). This
repository carries that fork as the `ggml` submodule. The fork includes the TTS
ops and backend behavior needed by Style-Bert-VITS2, Kokoro, and the other
models in this repository.

If you clone manually, initialize the submodule before building:

```bash
git submodule update --init --recursive
```

#### Build:

Build the default configuration:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Build with Metal on macOS:

```bash
cmake -B build-metal -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build-metal --config Release -j
```

Build with Vulkan:

```bash
cmake -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-vulkan --config Release -j
```

The CLI and server executables are emitted under the build directory's `bin`
folder, for example `./build/bin/tts-cli` and `./build/bin/tts-server`.

If you wish to install TTS.cpp with eSpeak NG phonemization support, first
[install eSpeak NG](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md).
Depending on your installation method, the path of the installed library will
vary. After identifying the installation path to eSpeak NG, it should contain
`./lib`, `./bin`, `./include`, and `./share` directories. Compile TTS.cpp with
eSpeak NG phonemization support by running the following in the repository base
directory:

```bash
export ESPEAK_INSTALL_DIR=/absolute/path/to/espeak-ng/dir
cmake -B build
cmake --build build --config Release
```

On Linux, you don't need to manually download or `export` anything. Our build system will automatically detect the development packages installed on your machine:

```bash
# Change `apt` and the package names to match your distro
sudo apt install build-essential cmake # Minimum requirements
sudo apt install git libespeak-ng-dev libsdl2-dev pkg-config # Optional requirements
cmake -B build
cmake --build build --config Release
```

### Usage

See the [CLI example readme](./examples/cli/README.md) for more details on its general usage.

Backend selection can be driven by CLI flags or environment variables:

```bash
./build/bin/tts-cli --backend metal --model-path /path/to/model.gguf --prompt "hello"
./build/bin/tts-server --backend vulkan --model-path /path/to/model.gguf

TTS_BACKEND=vulkan TTS_DEVICE=0 TTS_BACKEND_STRICT=1 \
  ./build-vulkan/bin/tts-server --model-path /path/to/model.gguf
```

Use `--host 0.0.0.0` when the server must be reachable from another machine on
the LAN. The default host is `127.0.0.1`.

### Style-Bert-VITS2 GGUF

Style-Bert-VITS2 is not exposed through the generic text `generate()` path. It
is exposed through server endpoints so an application can preserve its own text
frontend, phone/tone, tokenizer, and style-selection logic. The caller sends
exact phones, tones, BERT features, speaker/style IDs, and synthesis parameters
to TTS.cpp.

Download the current GGUF assets from the same repository. `voices/*.gguf`
contains Style-Bert-VITS2 TTS models. `frontend/style-bert-vits2-jp-bert.gguf`
is the required JP-BERT frontend feature extractor used by those TTS models; it
is not a standalone TTS model. The published JP-BERT artifact uses F16 `linear`
weights while keeping embeddings, conv, norm, and bias tensors F32.

```bash
hf download kevinzhow/style-bert-vits2-gguf \
  voices/jvnv-F1-jp-full-sdp.gguf \
  frontend/style-bert-vits2-jp-bert.gguf \
  --local-dir ./tmp/style-bert-vits2-gguf
```

Download every voice model that the caller may request. The JP-BERT GGUF is
shared, but each Style-Bert voice still needs its own `voices/*.gguf` decoder
model. For example, an application that can select Chu2 or Mai also needs:

```bash
hf download kevinzhow/style-bert-vits2-gguf \
  voices/chu2-full-sdp.gguf \
  voices/mai-full-sdp.gguf \
  --local-dir ./tmp/style-bert-vits2-gguf
```

Suggested local asset layout:

```text
tmp/style-bert-vits2-gguf/voices/*.gguf
tmp/style-bert-vits2-gguf/frontend/style-bert-vits2-jp-bert.gguf
```

Run separate decoder and JP-BERT servers:

```bash
# Style-Bert-VITS2 decoder/front-end graph server
TTS_BACKEND=vulkan TTS_DEVICE=0 TTS_BACKEND_STRICT=1 \
  ./build-vulkan/bin/tts-server \
  --model-path ./tmp/style-bert-vits2-gguf/voices \
  --default-model jvnv-F1-jp-full-sdp \
  --host 127.0.0.1 \
  --port 18102 \
  --backend vulkan

# JP-BERT feature server
TTS_BACKEND=vulkan TTS_DEVICE=0 TTS_BACKEND_STRICT=1 \
  ./build-vulkan/bin/tts-server \
  --model-path ./tmp/style-bert-vits2-gguf/frontend/style-bert-vits2-jp-bert.gguf \
  --host 127.0.0.1 \
  --port 18103 \
  --backend vulkan
```

For local macOS Metal runs, use the same commands with `build-metal` and
`--backend metal`.

`tts-server` can also load a directory of `.gguf` files. When a directory
contains multiple models, pass the desired model ID in each request's `model`
field or set `--default-model`.

Verify the loaded decoder models before wiring an application or sidecar to the
server:

```bash
curl -fsS http://127.0.0.1:18102/v1/models
```

That endpoint is the source of truth for model IDs accepted by
`/v1/style-bert-vits2/*` synthesis requests. If a request fails with
`Invalid Model: chu2-full-sdp`, then `chu2-full-sdp.gguf` was not loaded by the
decoder server. Download the matching GGUF file, restart the decoder server,
and re-check `/v1/models`. Do not treat this as a JP-BERT tokenizer, fallback,
or audio-quality problem.

Common built-in voice model IDs are `jvnv-F1-jp-full-sdp`,
`jvnv-F2-jp-full-sdp`, `jvnv-M1-jp-full-sdp`, `jvnv-M2-jp-full-sdp`,
`amitaro-full-sdp`, `koharune-ami-full-sdp`, `mao-full-sdp`,
`michinoku-airi-full-sdp`, `chu2-full-sdp`, `nise-full-sdp`,
`kanon-full-sdp`, `mai-full-sdp`, and `runa-full-sdp`.

#### Style-Bert-VITS2 Server Endpoints

The Style-Bert-specific routes are:

| Endpoint | Purpose |
| --- | --- |
| `/v1/style-bert-vits2/decode` | Decode `decoder_z` and `decoder_g` into audio |
| `/v1/style-bert-vits2/synthesize-latent` | Build alignment/latent flow from duration and prior tensors, then decode |
| `/v1/style-bert-vits2/synthesize-front` | Run Style-Bert front-end graph from `phone_ids`, `tone_ids`, `language_ids`, and `bert` |
| `/v1/style-bert-vits2/synthesize-symbols` | Convert symbolic `phones` and `tones` to IDs, then run the front-end graph |
| `/v1/style-bert-vits2/jp-bert/features` | Run JP-BERT from tokenizer `input_ids` and return base64 float32 features |

`synthesize-front` and `synthesize-symbols` accept Style-Bert controls such as
`speaker_id`, `style_id`, `style_weight`, `sdp_ratio`, `length_scale`,
`noise_scale`, `sdp_noise_scale`, `response_format`, and `return_alignment`.
Audio responses currently support `wav` and `aiff`.

The JP-BERT endpoint does not tokenize text. The application must use the same
Japanese tokenizer used for conversion and send `input_ids` to
`/v1/style-bert-vits2/jp-bert/features`. Missing tokenizer files are therefore
an application/sidecar setup issue, while `Invalid Model: <id>` is a decoder
server loading issue.

#### Style-Bert-VITS2 Accuracy and Performance

The Style-Bert Vulkan path defaults to accuracy-first settings. Unless you
explicitly opt into fast mode, the loader disables Vulkan F16 and cooperative
matrix paths for Style-Bert and JP-BERT:

```bash
# Accuracy-first default behavior; these are set automatically if unset.
GGML_VK_DISABLE_F16=1
GGML_VK_DISABLE_COOPMAT=1
GGML_VK_DISABLE_COOPMAT2=1

# Opt in only when profiling a known-good target device.
STYLE_BERT_VITS2_VULKAN_PRECISION=fast
STYLE_BERT_VITS2_JP_BERT_VULKAN_PRECISION=fast
```

Useful diagnostic and tuning environment variables:

| Variable | Use |
| --- | --- |
| `STYLE_BERT_VITS2_DEBUG_TIMINGS=1` | Print graph timing breakdowns for decoder, latent, flow, and front-end phases |
| `STYLE_BERT_VITS2_DEBUG_LOAD=1` | Print Style-Bert load details and Vulkan precision mode |
| `STYLE_BERT_VITS2_JP_BERT_DEBUG_LOAD=1` | Print JP-BERT load details and Vulkan precision mode |
| `STYLE_BERT_VITS2_ATTENTION_MODE=full` | Force the current full attention path |
| `STYLE_BERT_VITS2_FLOW_GROUP_SIZE=<n>` | Group flow reverse layers during profiling |
| `STYLE_BERT_VITS2_FLOW_FUSED=1` | Use the fused flow graph experiment |
| `STYLE_BERT_VITS2_METAL_TILED_CONV1D=1` | Enable Metal tiled conv1d path explicitly |

Leave experimental flow and attention knobs unset unless you are comparing
accuracy and RTF on a fixed model/input pair.

#### Style-Bert-VITS2 Conversion

The Style-Bert voice converter accepts either a Style-Bert model directory or
explicit model/config/style-vector paths:

```bash
PYTHONPATH=./py-gguf \
python3 ./py-gguf/convert_style_bert_vits2_to_gguf \
  --model-dir /path/to/style-bert-model-dir \
  --save-path ./tmp/style-bert-vits2-gguf/voices/example-full-sdp.gguf
```

JP-BERT conversion is handled separately:

```bash
PYTHONPATH=./py-gguf \
python3 ./py-gguf/convert_style_bert_vits2_jp_bert_to_gguf \
  --bert-dir /path/to/deberta-v2-large-japanese-char-wwm \
  --save-path ./tmp/style-bert-vits2-gguf/frontend/style-bert-vits2-jp-bert.gguf
```

### Quantization and Lower Precision Models

See the [quantization cli readme](./examples/quantize/README.md) for more
details on its general usage and behavior. Quantization and lower precision
conversion are supported for JP-BERT linear weights. The published
Style-Bert-VITS2 JP-BERT frontend artifact is generated with:

```bash
./build/bin/quantize \
  --model-path ./tmp/style-bert-vits2-gguf/frontend/style-bert-vits2-jp-bert-f32.gguf \
  --quantized-model-path ./tmp/style-bert-vits2-gguf/frontend/style-bert-vits2-jp-bert.gguf \
  --quantized-type F16 \
  --jp-bert-quantize-scope linear
```

Do not use `--jp-bert-quantize-scope all_weights` as the default Vulkan asset:
it can route norm inputs through F16 tensors and abort on backends that do not
implement F16 norm. Q8/Q4 JP-BERT recipes need separate audio-parity validation
before being used as defaults.

### Performance

For legacy model performance testing, see the
[performance battery readme](./examples/perf_battery/README.md).

For Style-Bert-VITS2, use `STYLE_BERT_VITS2_DEBUG_TIMINGS=1` and measure the
complete application-side RTF with the same frontend outputs, BERT features,
speaker, style, and synthesis settings. Vulkan and Metal timing should be
compared against the same GGUF files from `kevinzhow/style-bert-vits2-gguf`.

# License

Unless indicated otherwise, this repo is `MIT`-licensed.

To the extent required by law, parts derived from the models' original implementations retain their original `Apache-2.0` license. This may include hyperparameters and post-processing logic, but excludes our port to ggml and C++. This makes the resulting binary `Apache-2.0`-licensed if those models are compiled in.

If eSpeak NG support is enabled, the resulting binary is `GPL-3.0-or-later`-licensed.
