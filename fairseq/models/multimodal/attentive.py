from typing import List, Dict, Optional
from dataclasses import dataclass, field

from torch import Tensor

from fairseq.dataclass import ChoiceEnum
from fairseq.distributed import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.models import register_model, register_model_architecture

from .multimodal_transformer import (
    MultimodalTransformerConfig,
    MultimodalTransformerDecoder,
    MultimodalTransformerModel
)
from .encoder import ProjectionalMultimodalTransformerEncoder
from .attentive_module import (
    SerialAttentiveMultimodalTransformerDecoderLayer,
    ParallelAttentiveMultimodalTransformerDecoderLayer,
    FlatAttentiveMultimodalTransformerDecoderLayer,
    HierarchicalAttentiveMultimodalTransformerDecoderLayer,
)
from .transformer import (
    transformer_tiny_wu2021,
    transformer_tiny_fan2020,
    transformer_tiny_hirasawa2023,
)

ATTENTIVE_LAYERS = {
    "serial": SerialAttentiveMultimodalTransformerDecoderLayer,
    "parallel": ParallelAttentiveMultimodalTransformerDecoderLayer,
    "flat": FlatAttentiveMultimodalTransformerDecoderLayer,
    "hierarchical": HierarchicalAttentiveMultimodalTransformerDecoderLayer,
}
COMBINATION_CHOICES = ChoiceEnum(ATTENTIVE_LAYERS.keys())

@dataclass
class AttentiveMultimodalTransformerConfig(MultimodalTransformerConfig):
    combination: COMBINATION_CHOICES = field(
        default="hierarchical",
        metadata={
            "help": "strategy to combine multiple modalities"
        }
    )
    modality_weights: List[float] = field(
        default_factory=lambda: [1.0, 1.0],
        metadata={
            "help": "weight for each modality in the parallel decoder layer"
        }
    )
    dropnet: Optional[float] = field(
        default=None,
        metadata={
            "help": "probability to drop either text or feature modality"
        }
    )

class AttentiveMultimodalTransformerDecoder(MultimodalTransformerDecoder):
    def __init__(self, 
        cfg: AttentiveMultimodalTransformerConfig,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)

    def build_decoder_layer(self, cfg: AttentiveMultimodalTransformerConfig, no_encoder_attn=False):
        layer = ATTENTIVE_LAYERS[cfg.combination](cfg, no_encoder_attn)

        # copied following code from `transformer_decoder.TransformerDecoderBase.build_decoder_layer``
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
    
    def extract_multimodal_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        
        Changes from the base function:
            - load `feat` and `feat_padding_mask`
            - pass `feat` and `feat_padding_mask` to layers
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        feat: Optional[Tensor] = None
        feat_padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["densed_feat"]) > 0:
            feat = encoder_out["densed_feat"][0]
        if encoder_out is not None and len(encoder_out["feat_padding_mask"]) > 0:
            feat_padding_mask = encoder_out["feat_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                feat,
                feat_padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

@register_model("attentive_multimodal_transformer")
class AttentiveMultimodalTransformerModel(MultimodalTransformerModel):
    dataclass_cls = AttentiveMultimodalTransformerConfig
    encoder_cls = ProjectionalMultimodalTransformerEncoder
    decoder_cls = AttentiveMultimodalTransformerDecoder

@register_model_architecture(
    "attentive_multimodal_transformer",
    "attentive_multimodal_transformer_tiny_wu2021"
)
def attentive_multimodal_transformer_tiny_wu2021(args):
    transformer_tiny_wu2021(args)

@register_model_architecture(
    "attentive_multimodal_transformer",
    "attentive_multimodal_transformer_tiny_fan2020"
)
def attentive_multimodal_transformer_tiny_fan2020(args):
    transformer_tiny_fan2020(args)

@register_model_architecture(
    "attentive_multimodal_transformer",
    "attentive_multimodal_transformer_tiny_hirasawa2023"
)
def attentive_multimodal_transformer_tiny_hirasawa2023(args):
    transformer_tiny_hirasawa2023(args)
