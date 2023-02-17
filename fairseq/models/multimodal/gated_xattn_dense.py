from typing import List, Dict, Optional
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from fairseq.distributed import fsdp_wrap
from fairseq.modules import LayerNorm
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.models import register_model, register_model_architecture

from .multimodal_transformer import (
    MultimodalTransformerConfig,
    MultimodalTransformerDecoder,
    MultimodalTransformerModel
)
from .encoder import ProjectionalMultimodalTransformerEncoder
from .attentive_module import AttentiveMultimodalTransformerDecoderLayerBase

from .transformer import transformer_tiny_hirasawa2023

class GatedXattnDenseDecoderLayer(AttentiveMultimodalTransformerDecoderLayerBase):
    """
    Inspired by Flamingo model
    Alayrac et al., 2022. Flamingo - a Visual Language Model for Few-Shot Learning
    """
    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(cfg, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.xattn_layernorm = LayerNorm(self.embed_dim, export=cfg.export)
        self.xattn = self.build_feat_attention(self.embed_dim, cfg)
        self.alpha_xattn = nn.Parameter(torch.tensor([0.]))
        
        self.dense_layernorm = LayerNorm(cfg.decoder.embed_dim)
        self.dense1 = self.build_fc1(
            self.embed_dim,
            cfg.decoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.dense2 = self.build_fc2(
            cfg.decoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.alpha_dense = nn.Parameter(torch.tensor([0.]))

        self.tanh = nn.Tanh()

    def get_alpha(self, raw_alpha, x, unimodal_mask):
        bsz = x.shape[1]
        alpha = self.tanh(raw_alpha).repeat(bsz)
        alpha[unimodal_mask] = 0.

        return alpha.reshape(1, -1, 1)

    def forward(
        self,
        x,
        encoder_out: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        feat: Optional[Tensor] = None,
        feat_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[Tensor]] = None,
        prev_attn_state: Optional[List[Tensor]] = None,
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        if (prev_attn_state is not None) or (prev_self_attn_state is not None):
            raise NotImplementedError()
        
        unimodal_mask = feat_padding_mask.all(dim=-1)

        # xattn
        if self.normalize_before:
            x = self.xattn_layernorm(x)
        enc_feat_state, _ = self.xattn(
            query=x,
            key=feat,
            value=feat,
            key_padding_mask=feat_padding_mask,
            static_kv=True,
            need_weights=False,
        )
        enc_feat_state = self.dropout_module(enc_feat_state)
        # x[:,multimodal_mask,:] = x[:,multimodal_mask,:] + self.tanh(self.alpha_xattn) * enc_feat_state[:,multimodal_mask,:]
        x = x + self.get_alpha(self.alpha_xattn, x, unimodal_mask) * enc_feat_state
        if not self.normalize_before:
            x = self.xattn_layernorm(x)
        
        # dense
        if self.normalize_before:
            x = self.dense_layernorm(x)
        densed_x = self.dense1(x)
        densed_x = self.activation_fn(densed_x)
        densed_x = self.activation_dropout_module(densed_x)
        densed_x = self.dense2(densed_x)
        densed_x = self.dropout_module(densed_x)
        # x[:,multimodal_mask,:] = x[:,multimodal_mask,:] + self.tanh(self.alpha_dense) * densed_x[:,multimodal_mask,:]
        x = x + self.get_alpha(self.alpha_dense, x, unimodal_mask) * densed_x
        if not self.normalize_before:
            x = self.dense_layernorm(x)

        return super().forward(
            x = x,
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            prev_self_attn_state=prev_self_attn_state,
            prev_attn_state=prev_attn_state,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_attn=need_attn,
            need_head_weights=need_head_weights,
        )

@dataclass
class GatedXattnDenseTransformerConfig(MultimodalTransformerConfig):
    pass

class GatedXattnDenseTransformerDecoder(MultimodalTransformerDecoder):
    def __init__(self, 
        cfg: GatedXattnDenseTransformerConfig,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)

    def build_decoder_layer(self, cfg: GatedXattnDenseTransformerConfig, no_encoder_attn=False):
        layer = GatedXattnDenseDecoderLayer(cfg, no_encoder_attn)

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

@register_model("gated_xattn_dense")
class AttentiveMultimodalTransformerModel(MultimodalTransformerModel):
    dataclass_cls = GatedXattnDenseTransformerConfig
    encoder_cls = ProjectionalMultimodalTransformerEncoder
    decoder_cls = GatedXattnDenseTransformerDecoder

@register_model_architecture(
    "gated_xattn_dense",
    "gated_xattn_dense_tiny"
)
def gated_xattn_dense_tiny(args):
    transformer_tiny_hirasawa2023(args)
