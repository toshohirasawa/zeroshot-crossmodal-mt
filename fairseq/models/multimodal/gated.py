from typing import Optional, Dict, List

import torch
from torch import nn, Tensor

from fairseq.models import (
    register_model,
    register_model_architecture,
)

from .encoder import ProjectionalMultimodalTransformerEncoder
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

class GatedMultimodalTransformerDecoder(MultimodalTransformerDecoder):
    def __init__(
        self,
        cfg: MultimodalTransformerConfig,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None
    ):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)

        embed_dim = self.cfg.decoder.embed_dim

        self.gate_sigmoid = nn.Sigmoid()
        self.gate_dense = nn.Linear(2 * embed_dim, embed_dim)

    def extract_multimodal_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        enc_padding_mask = encoder_out['encoder_padding_mask'][0]
        enc_out = mask_pad_elements(
            encoder_out['encoder_out'][0],
            enc_padding_mask,
        ) # (time_enc, bsz, embed_dim)

        # use the averaged element in each feature
        feat_padding_mask = encoder_out['feat_padding_mask'][0][:, :1]
        feat = mask_pad_elements(
             encoder_out['densed_feat'][0].mean(dim=0, keepdim=True),
             feat_padding_mask,
        ).expand(enc_out.size()) # (time_enc, bsz, embed_dim)

        # append the global feat to each timestep and computing the lambda
        multimodal_state = torch.cat([enc_out, feat], dim=-1)
        gate_lambda = self.gate_sigmoid(self.gate_dense(multimodal_state))

        # mask-out lambda at positions with padded features
        gate_lambda = mask_pad_elements(gate_lambda, feat_padding_mask.expand(enc_padding_mask.size()))

        enc_out = enc_out * (1 - gate_lambda) + feat * gate_lambda

        encoder_out['encoder_out'][0] = enc_out

        return self.extract_features_scriptable(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

@register_model("gated_multimodal_transformer")
class GatedMultimodalTransformerModel(MultimodalTransformerModel):
    """
    Gated fusion model from
    Wu et al., 2021. Good for Misconceived Reasons - An Empirical Revisiting on the Need for Visual Context in Multimodal Machine Translation

    - Vision model: ResNet-50 CNN
    - Feature type: average-pooled
    """
    dataclass_cls = MultimodalTransformerConfig
    encoder_cls = ProjectionalMultimodalTransformerEncoder
    decoder_cls = GatedMultimodalTransformerDecoder

@register_model_architecture("gated_multimodal_transformer", "gated_multimodal_transformer_tiny_wu2021")
def gated_multimodal_transformer_tiny_wu2021(args):
    return transformer_tiny_wu2021(args)

@register_model_architecture("gated_multimodal_transformer", "gated_multimodal_transformer_tiny_fan2020")
def gated_multimodal_transformer_tiny_fan2020(args):
    return transformer_tiny_fan2020(args)

@register_model_architecture("gated_multimodal_transformer", "gated_multimodal_transformer_tiny_hirasawa2023")
def gated_multimodal_transformer_tiny_hirasawa2023(args):
    return transformer_tiny_hirasawa2023(args)
