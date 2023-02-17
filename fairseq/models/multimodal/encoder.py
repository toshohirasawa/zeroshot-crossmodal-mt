from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from fairseq.modules import LayerNorm

from .multimodal_transformer import MultimodalTransformerEncoder

class ProjectionalMultimodalTransformerEncoder(MultimodalTransformerEncoder):

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        return_fc=False
    ):
        super().__init__(cfg, dictionary, embed_tokens, return_fc)
        self.embed_dim = self.cfg.encoder.embed_dim

        self.projection_layer_norm = LayerNorm(self.feat_channel)
        self.projection = nn.Linear(self.feat_channel, self.embed_dim, bias=True)

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[Tensor] = None,
        feat: Optional[Tensor] = None,
        feat_padding_mask: Optional[Tensor] = None,
        tgt_lang_tok: Optional[Tensor] = None,
    ):
        enc_out = self.forward_projection_scriptable(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=token_embeddings,
            feat=feat, 
            feat_padding_mask=feat_padding_mask,
            tgt_lang_tok=tgt_lang_tok
        )

        return enc_out
    
    def forward_projection_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[Tensor] = None,
        feat: Optional[Tensor] = None,
        feat_padding_mask: Optional[Tensor] = None,
        tgt_lang_tok: Optional[Tensor] = None,
    ):
        enc_out = self.forward_passthrough_scriptable(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=token_embeddings,
            feat=feat, 
            feat_padding_mask=feat_padding_mask,
            tgt_lang_tok=tgt_lang_tok
        )

        densed_feat = self.projection_layer_norm(feat)
        densed_feat = self.projection(densed_feat)

        enc_out['densed_feat'] = [densed_feat]

        return enc_out

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        ordered_encoder_out = super().reorder_encoder_out(
            encoder_out=encoder_out,
            new_order=new_order
        )

        if len(encoder_out["densed_feat"]) == 0:
            new_densed_feat = []
        else:
            new_densed_feat = [encoder_out["densed_feat"][0].index_select(1, new_order)]

        ordered_encoder_out.update({
            "densed_feat": new_densed_feat
        })

        return ordered_encoder_out