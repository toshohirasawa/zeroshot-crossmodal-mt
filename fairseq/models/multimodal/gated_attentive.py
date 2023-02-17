from typing import Optional, Dict, List
from dataclasses import dataclass, field

import torch
from torch import nn, Tensor

from fairseq.modules import (
    LayerNorm,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)

from .utils import MaskedMultiheadAttention
from .encoder import ProjectionalMultimodalTransformerEncoder
from .autoencoder import AutoencoderMultimodalTransformerEncoder
from .transformer import (
    transformer_tiny_wu2021,
    transformer_tiny_fan2020,
    transformer_tiny_hirasawa2023,
)
from .multimodal_transformer import (
    MultimodalTransformerConfig,
    MultimodalTransformerModel,
    MultimodalTransformerDecoder,
)
from .utils import mask_pad_elements

@dataclass
class GatedAttentiveMultimodalTransformerConfig(MultimodalTransformerConfig):
    pass

class GatedAttentiveMultimodalTransformerDecoder(MultimodalTransformerDecoder):
    def __init__(
        self,
        cfg: GatedAttentiveMultimodalTransformerConfig,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None
    ):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)

        embed_dim = self.cfg.decoder.embed_dim

        self.encoder_feat_attn = self.build_selective_attention(cfg)
        self.encoder_feat_attn_layer_norm = LayerNorm(self.embed_dim)

        self.gate_sigmoid = nn.Sigmoid()
        self.gate_dense = nn.Linear(2 * embed_dim, 1)

    def build_selective_attention(self, cfg) -> MaskedMultiheadAttention:
        return MaskedMultiheadAttention(
            self.embed_dim,
            cfg.decoder.attention_heads,
            kdim=self.embed_dim,
            vdim=self.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=cfg.quant_noise.pq,
            qn_block_size=cfg.quant_noise.pq_block_size,
            xformers_att_config=cfg.decoder.xformers_att_config,
            keep_query=True,
        )

    def residual_connection(self, x, residual):
        return residual + x
    
    def gate(self, enc_out, enc_feat_state):
        multimodal_state = torch.cat([enc_out, enc_feat_state], dim=-1)
        gate_lambda = self.gate_sigmoid(self.gate_dense(multimodal_state))

        gate_lambda = gate_lambda.expand(-1, -1, enc_out.shape[-1])
        
        return gate_lambda

    def extract_multimodal_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # encoder hidden states
        enc_padding_mask = encoder_out['encoder_padding_mask'][0]
        enc_out = mask_pad_elements(
            encoder_out['encoder_out'][0],
            enc_padding_mask,
        ) # (time_enc, bsz, embed_dim)

        # vision features
        feat_padding_mask = encoder_out['feat_padding_mask'][0]
        feat = mask_pad_elements(
             encoder_out['densed_feat'][0],
             feat_padding_mask,
        ) # (n_channel, bsz, embed_dim)
        
        # enc_out to feat attention
        residual = enc_out
        enc_feat_state, _ = self.encoder_feat_attn(
            query = self.encoder_feat_attn_layer_norm(enc_out),
            key = feat,
            value = feat,
            key_padding_mask = feat_padding_mask,
            incremental_state = incremental_state,
            static_kv = True,
        )
        enc_feat_state = self.dropout_module(enc_feat_state)
        enc_feat_state = self.residual_connection(enc_feat_state, residual)

        # append the global feat to each timestep and computing the lambda
        gate_lambda = self.gate(enc_out, enc_feat_state)

        # mask-out lambda at positions with padded features
        all_feat_padding_mask = feat_padding_mask.all(dim=-1).unsqueeze(-1)
        gate_lambda = mask_pad_elements(gate_lambda, all_feat_padding_mask.repeat(1, gate_lambda.shape[0]))

        enc_out = enc_out * (1 - gate_lambda) + enc_feat_state * gate_lambda

        encoder_out['encoder_out'][0] = enc_out

        return self.extract_features_scriptable(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

@register_model("gated_attentive")
class GatedAttentiveMultimodalTransformerModel(MultimodalTransformerModel):
    dataclass_cls = GatedAttentiveMultimodalTransformerConfig
    encoder_cls = ProjectionalMultimodalTransformerEncoder
    decoder_cls = GatedAttentiveMultimodalTransformerDecoder

@register_model_architecture(
    "gated_attentive", 
    "gated_attentive_tiny"
)
def gated_attentive_tiny(args):
    return transformer_tiny_hirasawa2023(args)
