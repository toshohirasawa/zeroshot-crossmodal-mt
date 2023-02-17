
from typing import Optional, Dict, List

import torch
from torch import Tensor

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer.transformer_base import TransformerModelBase

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

class ConcatMultimodalTransformerDecoder(MultimodalTransformerDecoder):
    def __init__(
        self, 
        cfg: MultimodalTransformerConfig, 
        dictionary, 
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None
    ):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)

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
        feat (LongTensor): vectorized features in the shape of (batch_size, feat_dim)
        """
        encoder_out['encoder_out'][0] = torch.cat([
            encoder_out['encoder_out'][0],
            encoder_out['densed_feat'][0],
        ], dim=0)

        encoder_out['encoder_padding_mask'][0] = torch.cat([
            encoder_out['encoder_padding_mask'][0],
            encoder_out['feat_padding_mask'][0],
        ], dim=-1)

        return self.extract_features_scriptable(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

@register_model("concat_multimodal_transformer")
class ConcatMultimodalTransformerModel(MultimodalTransformerModel):
    dataclass_cls = MultimodalTransformerConfig
    encoder_cls = ProjectionalMultimodalTransformerEncoder
    decoder_cls = ConcatMultimodalTransformerDecoder

@register_model_architecture("concat_multimodal_transformer", "concat_multimodal_transformer_tiny")
def concat_multimodal_transformer_tiny(args):
    return transformer_tiny_hirasawa2023(args)
