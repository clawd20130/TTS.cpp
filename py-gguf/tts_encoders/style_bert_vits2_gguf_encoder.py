import json
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch

from .tensor_util import get_regularized_weight
from .tts_encoder import TTSEncoder


STYLE_BERT_VITS2_ARCHITECTURE = "style-bert-vits2"


def _compact_encoder_tensor_name(name: str) -> str:
    parts = name.split(".")
    if parts[0] == "spk_emb_linear" and len(parts) == 2:
        return f"spk.{'w' if parts[1] == 'weight' else 'b'}"
    if parts[0] == "attn_layers" and len(parts) >= 3:
        layer = parts[1]
        if parts[2] == "emb_rel_k":
            return f"al.{layer}.rk"
        if parts[2] == "emb_rel_v":
            return f"al.{layer}.rv"
        if len(parts) == 4 and parts[2].startswith("conv_"):
            proj = parts[2].removeprefix("conv_")
            leaf = "w" if parts[3] == "weight" else "b"
            return f"al.{layer}.{proj}.{leaf}"
    if parts[0] == "ffn_layers" and len(parts) == 4:
        layer = parts[1]
        conv = "c1" if parts[2] == "conv_1" else "c2"
        leaf = "w" if parts[3] == "weight" else "b"
        return f"ffn.{layer}.{conv}.{leaf}"
    if parts[0] in {"norm_layers_1", "norm_layers_2"} and len(parts) == 3:
        norm = "n1" if parts[0] == "norm_layers_1" else "n2"
        leaf = "g" if parts[2] == "gamma" else "b"
        return f"{norm}.{parts[1]}.{leaf}"
    raise ValueError(f"Unhandled Style-Bert encoder tensor name: {name}")


def _compact_flow_tensor_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "flows":
        raise ValueError(f"Unhandled Style-Bert flow tensor name: {name}")
    raw_index = int(parts[1])
    if raw_index % 2:
        raise ValueError(f"Unexpected parameter on Style-Bert Flip layer: {name}")
    layer = raw_index // 2
    rest = parts[2:]
    if rest[0] in {"pre", "post"} and len(rest) == 2:
        leaf = "w" if rest[1] == "weight" else "b"
        return f"{layer}.{rest[0]}.{leaf}"
    if rest[0] == "enc":
        return f"{layer}.enc.{_compact_encoder_tensor_name('.'.join(rest[1:]))}"
    raise ValueError(f"Unhandled Style-Bert flow tensor name: {name}")


class StyleBertVits2Encoder(TTSEncoder):
    """Encode Style-Bert-VITS2 tensors for the GGML runtime.

    The first supported target is the neural decoder/vocoder subgraph. It is
    intentionally kept under the final Style-Bert architecture so the same GGUF
    can be extended with encoder, duration, flow, and tokenizer tensors.
    """

    def __init__(
        self,
        model_path: Path | str,
        *,
        source_model_path: Path | str,
        config_path: Path | str,
        style_vec_path: Path | str,
        device: str = "cpu",
    ) -> None:
        super().__init__(model_path=model_path, architecture=STYLE_BERT_VITS2_ARCHITECTURE)
        self.source_model_path = Path(source_model_path)
        self.config_path = Path(config_path)
        self.style_vec_path = Path(style_vec_path)
        self.device = device
        self.repo_id = str(self.source_model_path)
        self._tts_model = None
        self._net_g = None
        self._config: dict[str, Any] | None = None

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            self._config = json.loads(self.config_path.read_text(encoding="utf-8"))
        return self._config

    @property
    def tts_model(self) -> Any:
        if self._tts_model is None:
            from style_bert_vits2.tts_model import TTSModel

            self._tts_model = TTSModel(
                model_path=self.source_model_path,
                config_path=self.config_path,
                style_vec_path=self.style_vec_path,
                device=self.device,
            )
            self._tts_model.load()
        return self._tts_model

    @property
    def net_g(self) -> Any:
        if self._net_g is None:
            self._net_g = self.tts_model.net_g
            if self._net_g is None:
                raise RuntimeError("Style-Bert-VITS2 TTSModel did not expose net_g after load.")
        return self._net_g

    def prepare_tensors(self) -> None:
        self.prepare_embedding_tensors()
        self.prepare_text_encoder_input_tensors()
        self.prepare_text_encoder_encoder_tensors()
        self.prepare_duration_predictor_tensors()
        self.prepare_stochastic_duration_predictor_tensors()
        self.prepare_flow_tensors()
        self.prepare_decoder_tensors()

    def prepare_embedding_tensors(self) -> None:
        if hasattr(self.net_g, "emb_g"):
            self.set_tensor("style_bert_vits2.speaker_embedding.weight", self.net_g.emb_g.weight)
        style_vectors = np.load(self.style_vec_path).astype(np.float32)
        self.set_tensor("style_bert_vits2.style_vectors", style_vectors)

    def prepare_text_encoder_input_tensors(self) -> None:
        base = "style_bert_vits2.text_encoder"
        enc_p = self.net_g.enc_p
        self.set_tensor(f"{base}.token_embedding.weight", enc_p.emb.weight)
        self.set_tensor(f"{base}.tone_embedding.weight", enc_p.tone_emb.weight)
        self.set_tensor(f"{base}.language_embedding.weight", enc_p.language_emb.weight)
        self.set_tensor(f"{base}.bert_proj.weight", enc_p.bert_proj.weight)
        self.set_tensor(f"{base}.bert_proj.bias", enc_p.bert_proj.bias)
        if hasattr(enc_p, "ja_bert_proj"):
            self.set_tensor(f"{base}.ja_bert_proj.weight", enc_p.ja_bert_proj.weight)
            self.set_tensor(f"{base}.ja_bert_proj.bias", enc_p.ja_bert_proj.bias)
        if hasattr(enc_p, "en_bert_proj"):
            self.set_tensor(f"{base}.en_bert_proj.weight", enc_p.en_bert_proj.weight)
            self.set_tensor(f"{base}.en_bert_proj.bias", enc_p.en_bert_proj.bias)
        self.set_tensor(f"{base}.style_proj.weight", enc_p.style_proj.weight)
        self.set_tensor(f"{base}.style_proj.bias", enc_p.style_proj.bias)
        self.set_tensor(f"{base}.proj.weight", enc_p.proj.weight)
        self.set_tensor(f"{base}.proj.bias", enc_p.proj.bias)

    def prepare_text_encoder_encoder_tensors(self) -> None:
        base = "style_bert_vits2.te.enc"
        for name, param in self.net_g.enc_p.encoder.named_parameters():
            self.set_tensor(f"{base}.{_compact_encoder_tensor_name(name)}", param)

    def prepare_duration_predictor_tensors(self) -> None:
        base = "style_bert_vits2.duration_predictor"
        dp = self.net_g.dp
        self.set_tensor(f"{base}.conv_1.weight", dp.conv_1.weight)
        self.set_tensor(f"{base}.conv_1.bias", dp.conv_1.bias)
        self.set_tensor(f"{base}.norm_1.gamma", dp.norm_1.gamma)
        self.set_tensor(f"{base}.norm_1.beta", dp.norm_1.beta)
        self.set_tensor(f"{base}.conv_2.weight", dp.conv_2.weight)
        self.set_tensor(f"{base}.conv_2.bias", dp.conv_2.bias)
        self.set_tensor(f"{base}.norm_2.gamma", dp.norm_2.gamma)
        self.set_tensor(f"{base}.norm_2.beta", dp.norm_2.beta)
        self.set_tensor(f"{base}.proj.weight", dp.proj.weight)
        self.set_tensor(f"{base}.proj.bias", dp.proj.bias)
        if hasattr(dp, "cond"):
            self.set_tensor(f"{base}.cond.weight", dp.cond.weight)
            self.set_tensor(f"{base}.cond.bias", dp.cond.bias)

    def prepare_stochastic_duration_predictor_tensors(self) -> None:
        if not hasattr(self.net_g, "sdp"):
            return
        base = "style_bert_vits2.sdp"
        for name, param in self.net_g.sdp.named_parameters():
            self.set_tensor(f"{base}.{name}", param)

    def prepare_flow_tensors(self) -> None:
        base = "style_bert_vits2.fl"
        for name, param in self.net_g.flow.named_parameters():
            self.set_tensor(f"{base}.{_compact_flow_tensor_name(name)}", param)

    def prepare_decoder_tensors(self) -> None:
        base = "style_bert_vits2.decoder"
        modules = {name: module for name, module in self.net_g.dec.named_modules()}
        for name, param in self.net_g.dec.named_parameters():
            parts = name.split(".")
            if parts[-1] == "weight_v":
                continue
            if parts[-1] == "weight_g":
                param = get_regularized_weight(modules, name)
                parts[-1] = "weight"
            self.set_tensor(f"{base}.{'.'.join(parts)}", param)

    def prepare_metadata(self) -> None:
        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()
        self.metadata = gguf.Metadata.load(None, None, STYLE_BERT_VITS2_ARCHITECTURE, total_params)
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(total_params, shared_params, expert_params, expert_count)
        self.set_type()
        self.metadata.set_gguf_meta_model(self.gguf_writer)
        self.set_gguf_parameters()
        self.gguf_writer.add_file_type(gguf.LlamaFileType.ALL_F32)
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    def set_type(self) -> None:
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)

    def set_gguf_parameters(self) -> None:
        model_config = self.config["model"]
        data_config = self.config["data"]
        decoder = self.net_g.dec
        arch = STYLE_BERT_VITS2_ARCHITECTURE

        self.gguf_writer.add_uint32(f"{arch}.decoder_only", 1)
        self.gguf_writer.add_uint32(f"{arch}.sample_rate", int(data_config["sampling_rate"]))
        self.gguf_writer.add_uint32(f"{arch}.inter_channels", int(model_config["inter_channels"]))
        self.gguf_writer.add_uint32(f"{arch}.hidden_channels", int(model_config["hidden_channels"]))
        self.gguf_writer.add_uint32(f"{arch}.filter_channels", int(model_config["filter_channels"]))
        self.gguf_writer.add_uint32(f"{arch}.n_heads", int(model_config["n_heads"]))
        self.gguf_writer.add_uint32(f"{arch}.text_encoder.n_layers", int(model_config["n_layers"]))
        self.gguf_writer.add_uint32(f"{arch}.text_encoder.kernel", int(model_config["kernel_size"]))
        self.gguf_writer.add_uint32(f"{arch}.text_encoder.window_size", 4)
        self.gguf_writer.add_uint32(f"{arch}.text_encoder.cond_layer_idx", int(getattr(self.net_g.enc_p.encoder, "cond_layer_idx", 2)))
        self.gguf_writer.add_uint32(f"{arch}.gin_channels", int(model_config["gin_channels"]))
        self.gguf_writer.add_uint32(f"{arch}.jp_extra", 1 if str(self.config.get("version", "")).endswith("JP-Extra") else 0)
        self.gguf_writer.add_uint32(f"{arch}.duration_predictor.filter_channels", int(self.net_g.dp.filter_channels))
        self.gguf_writer.add_uint32(f"{arch}.duration_predictor.kernel", int(self.net_g.dp.kernel_size))
        self.gguf_writer.add_uint32(f"{arch}.duration_predictor.padding", int(self.net_g.dp.kernel_size // 2))
        if hasattr(self.net_g, "sdp"):
            self.gguf_writer.add_uint32(f"{arch}.sdp.n_flows", int(self.net_g.sdp.n_flows))
            self.gguf_writer.add_uint32(f"{arch}.sdp.n_layers", 3)
            self.gguf_writer.add_uint32(f"{arch}.sdp.kernel", int(self.net_g.sdp.kernel_size))
            self.gguf_writer.add_uint32(f"{arch}.sdp.num_bins", 10)
        self.gguf_writer.add_uint32(f"{arch}.flow.use_transformer", 1)
        self.gguf_writer.add_uint32(f"{arch}.flow.n_flows", int(self.net_g.flow.n_flows))
        self.gguf_writer.add_uint32(f"{arch}.flow.n_layers", int(self.net_g.flow.n_layers))
        self.gguf_writer.add_uint32(f"{arch}.flow.kernel", int(self.net_g.flow.kernel_size))
        self.gguf_writer.add_uint32(f"{arch}.flow.window_size", 4)
        self.gguf_writer.add_uint32(f"{arch}.flow.cond_layer_idx", 2)
        self.gguf_writer.add_uint32(f"{arch}.upsample_initial_channel", int(model_config["upsample_initial_channel"]))
        self.gguf_writer.add_uint32(f"{arch}.decoder.num_upsamples", int(decoder.num_upsamples))
        self.gguf_writer.add_uint32(f"{arch}.decoder.num_kernels", int(decoder.num_kernels))
        self.gguf_writer.add_uint32(f"{arch}.decoder.resblock", int(model_config["resblock"]))
        self.gguf_writer.add_uint32(f"{arch}.decoder.conv_pre.padding", int(decoder.conv_pre.padding[0]))
        self.gguf_writer.add_uint32(f"{arch}.decoder.conv_pre.kernel", int(decoder.conv_pre.kernel_size[0]))
        self.gguf_writer.add_uint32(f"{arch}.decoder.conv_post.padding", int(decoder.conv_post.padding[0]))
        self.gguf_writer.add_uint32(f"{arch}.decoder.conv_post.kernel", int(decoder.conv_post.kernel_size[0]))

        for index, upsample in enumerate(decoder.ups):
            self.gguf_writer.add_uint32(f"{arch}.decoder.ups.{index}.stride", int(upsample.stride[0]))
            self.gguf_writer.add_uint32(f"{arch}.decoder.ups.{index}.padding", int(upsample.padding[0]))
            self.gguf_writer.add_uint32(f"{arch}.decoder.ups.{index}.kernel", int(upsample.kernel_size[0]))

        for block_index, resblock in enumerate(decoder.resblocks):
            for conv_index, conv in enumerate(resblock.convs1):
                self.gguf_writer.add_uint32(
                    f"{arch}.decoder.resblocks.{block_index}.convs1.{conv_index}.padding",
                    int(conv.padding[0]),
                )
                self.gguf_writer.add_uint32(
                    f"{arch}.decoder.resblocks.{block_index}.convs1.{conv_index}.dilation",
                    int(conv.dilation[0]),
                )
                self.gguf_writer.add_uint32(
                    f"{arch}.decoder.resblocks.{block_index}.convs1.{conv_index}.kernel",
                    int(conv.kernel_size[0]),
                )
            for conv_index, conv in enumerate(resblock.convs2):
                self.gguf_writer.add_uint32(
                    f"{arch}.decoder.resblocks.{block_index}.convs2.{conv_index}.padding",
                    int(conv.padding[0]),
                )
                self.gguf_writer.add_uint32(
                    f"{arch}.decoder.resblocks.{block_index}.convs2.{conv_index}.dilation",
                    int(conv.dilation[0]),
                )
                self.gguf_writer.add_uint32(
                    f"{arch}.decoder.resblocks.{block_index}.convs2.{conv_index}.kernel",
                    int(conv.kernel_size[0]),
                )
