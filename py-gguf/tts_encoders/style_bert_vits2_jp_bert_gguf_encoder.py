import json
from pathlib import Path
from typing import Any

import gguf
import torch

from .tts_encoder import TTSEncoder


STYLE_BERT_VITS2_JP_BERT_ARCHITECTURE = "style-bert-vits2-jp-bert"


def _compact_deberta_tensor_name(name: str) -> str | None:
    if name == "deberta.embeddings.position_ids":
        return None
    if name.startswith("cls.predictions."):
        return None

    prefix = "deberta."
    if not name.startswith(prefix):
        raise ValueError(f"Unhandled Style-Bert JP BERT tensor name: {name}")

    parts = name[len(prefix):].split(".")
    if parts[:2] == ["embeddings", "word_embeddings"] and parts[2] == "weight":
        return "emb.word.weight"
    if parts[:2] == ["embeddings", "LayerNorm"]:
        return f"emb.norm.{parts[2]}"
    if parts[0] == "encoder" and parts[1] == "rel_embeddings" and parts[2] == "weight":
        return "enc.rel_embeddings.weight"
    if parts[0] == "encoder" and parts[1] == "LayerNorm":
        return f"enc.norm.{parts[2]}"
    if parts[:3] == ["encoder", "conv", "conv"]:
        return f"enc.conv.conv.{parts[3]}"
    if parts[:3] == ["encoder", "conv", "LayerNorm"]:
        return f"enc.conv.norm.{parts[3]}"
    if parts[0] == "encoder" and parts[1] == "layer":
        layer = parts[2]
        rest = parts[3:]
        if rest[:2] == ["attention", "self"] and rest[2].endswith("_proj"):
            proj = rest[2].removesuffix("_proj")
            return f"layers.{layer}.attn.self.{proj}.{rest[3]}"
        if rest[:2] == ["attention", "output"] and rest[2] == "dense":
            return f"layers.{layer}.attn.out.dense.{rest[3]}"
        if rest[:2] == ["attention", "output"] and rest[2] == "LayerNorm":
            return f"layers.{layer}.attn.out.norm.{rest[3]}"
        if rest[0] == "intermediate" and rest[1] == "dense":
            return f"layers.{layer}.intermediate.dense.{rest[2]}"
        if rest[0] == "output" and rest[1] == "dense":
            return f"layers.{layer}.output.dense.{rest[2]}"
        if rest[0] == "output" and rest[1] == "LayerNorm":
            return f"layers.{layer}.output.norm.{rest[2]}"

    raise ValueError(f"Unhandled Style-Bert JP BERT tensor name: {name}")


class StyleBertVits2JpBertEncoder(TTSEncoder):
    """Encode the JP DeBERTa feature extractor used by Style-Bert-VITS2."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        bert_dir: Path | str,
        max_layers: int | None = None,
    ) -> None:
        super().__init__(model_path=model_path, architecture=STYLE_BERT_VITS2_JP_BERT_ARCHITECTURE)
        self.bert_dir = Path(bert_dir)
        self.repo_id = str(self.bert_dir)
        self.max_layers = max_layers
        self._config: dict[str, Any] | None = None
        self._state_dict: dict[str, torch.Tensor] | None = None

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            self._config = json.loads((self.bert_dir / "config.json").read_text(encoding="utf-8"))
        return self._config

    @property
    def state_dict(self) -> dict[str, torch.Tensor]:
        if self._state_dict is None:
            self._state_dict = torch.load(self.bert_dir / "pytorch_model.bin", map_location="cpu")
        return self._state_dict

    @property
    def layer_count(self) -> int:
        configured = int(self.config["num_hidden_layers"])
        if self.max_layers is None:
            return configured
        if self.max_layers <= 0 or self.max_layers > configured:
            raise ValueError(f"max_layers must be between 1 and {configured}; got {self.max_layers}.")
        return int(self.max_layers)

    def prepare_tensors(self) -> None:
        base = STYLE_BERT_VITS2_JP_BERT_ARCHITECTURE
        allowed_layers = self.layer_count
        for name, tensor in self.state_dict.items():
            compact = _compact_deberta_tensor_name(name)
            if compact is None:
                continue
            if compact.startswith("layers."):
                layer = int(compact.split(".", 2)[1])
                if layer >= allowed_layers:
                    continue
            self.set_tensor(f"{base}.{compact}", tensor)

    def prepare_metadata(self) -> None:
        arch = STYLE_BERT_VITS2_JP_BERT_ARCHITECTURE
        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()
        self.metadata = gguf.Metadata.load(None, None, arch, total_params)
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(total_params, shared_params, expert_params, expert_count)
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)
        self.metadata.set_gguf_meta_model(self.gguf_writer)
        self.set_gguf_parameters()
        self.add_tokenizer_metadata()
        self.gguf_writer.add_file_type(gguf.LlamaFileType.ALL_F32)
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    def set_gguf_parameters(self) -> None:
        arch = STYLE_BERT_VITS2_JP_BERT_ARCHITECTURE
        config = self.config
        self.gguf_writer.add_vocab_size(int(config["vocab_size"]))
        self.gguf_writer.add_uint32(f"{arch}.hidden_size", int(config["hidden_size"]))
        self.gguf_writer.add_uint32(f"{arch}.intermediate_size", int(config["intermediate_size"]))
        self.gguf_writer.add_uint32(f"{arch}.layers", self.layer_count)
        self.gguf_writer.add_uint32(f"{arch}.attn_heads", int(config["num_attention_heads"]))
        self.gguf_writer.add_uint32(f"{arch}.head_size", int(config["hidden_size"]) // int(config["num_attention_heads"]))
        self.gguf_writer.add_uint32(f"{arch}.max_position_embeddings", int(config["max_position_embeddings"]))
        self.gguf_writer.add_uint32(f"{arch}.position_buckets", int(config["position_buckets"]))
        self.gguf_writer.add_int32(f"{arch}.max_relative_positions", int(config["max_relative_positions"]))
        self.gguf_writer.add_uint32(f"{arch}.type_vocab_size", int(config.get("type_vocab_size", 0)))
        self.gguf_writer.add_uint32(f"{arch}.pad_token_id", int(config["pad_token_id"]))
        self.gguf_writer.add_float32(f"{arch}.layer_norm_eps", float(config["layer_norm_eps"]))
        self.gguf_writer.add_string(f"{arch}.hidden_act", str(config["hidden_act"]))
        self.gguf_writer.add_bool(f"{arch}.relative_attention", bool(config["relative_attention"]))
        self.gguf_writer.add_bool(f"{arch}.position_biased_input", bool(config["position_biased_input"]))
        self.gguf_writer.add_bool(f"{arch}.share_att_key", bool(config["share_att_key"]))
        self.gguf_writer.add_array(f"{arch}.pos_att_type", list(config.get("pos_att_type", [])))
        self.gguf_writer.add_uint32(f"{arch}.feature_hidden_state_offset", 2)

    def add_tokenizer_metadata(self) -> None:
        from transformers import BertJapaneseTokenizer

        arch = STYLE_BERT_VITS2_JP_BERT_ARCHITECTURE
        tokenizer = BertJapaneseTokenizer.from_pretrained(self.bert_dir)
        ordered_vocab = [token for token, _ in sorted(tokenizer.vocab.items(), key=lambda item: item[1])]
        self.gguf_writer.add_tokenizer_model("bert-japanese")
        self.gguf_writer.add_tokenizer_pre("style-bert-vits2-jp")
        self.gguf_writer.add_token_list(ordered_vocab)
        self.gguf_writer.add_pad_token_id(int(tokenizer.pad_token_id))
        self.gguf_writer.add_cls_token_id(int(tokenizer.cls_token_id))
        self.gguf_writer.add_sep_token_id(int(tokenizer.sep_token_id))
        self.gguf_writer.add_unk_token_id(int(tokenizer.unk_token_id))
        self.gguf_writer.add_mask_token_id(int(tokenizer.mask_token_id))
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(False)
        self.gguf_writer.add_string(f"{arch}.tokenizer_class", tokenizer.__class__.__name__)
        self.gguf_writer.add_string(f"{arch}.word_tokenizer_type", str(getattr(tokenizer, "word_tokenizer_type", "")))
        self.gguf_writer.add_string(f"{arch}.subword_tokenizer_type", str(getattr(tokenizer, "subword_tokenizer_type", "")))
