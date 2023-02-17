from typing import List, Dict, Optional
from dataclasses import dataclass, field
from fairseq.dataclass import ChoiceEnum

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
from .multimodal_transformer import (
    MultimodalTransformerConfig,
    MultimodalTransformerModel,
    MultimodalTransformerEncoder,
    MultimodalTransformerDecoder,
)
from .transformer import (
    transformer_tiny_hirasawa2023,
    transformer_small_hirasawa2023,
)

class StaticProjectionLayer(nn.Module):
    def __init__(self, feat_channel, embed_dim, **kwargs) -> None:
        super().__init__()
        self.dense = nn.Linear(feat_channel, embed_dim)
    
    def forward(self, feat, **kwgargs):
        return self.dense(feat)

class SeparateProjectionLayer(nn.Module):
    def __init__(self, feat_channel, embed_dim, tgt_lang_toks: List[int], **kwargs) -> None:
        # this assertion will throw an exception for the multimodal_translation tasks
        assert (tgt_lang_toks is not None) and (len(tgt_lang_toks) > 0)

        super().__init__()

        self.tgt_lang_toks = tgt_lang_toks
        self.embed_dim = embed_dim

        self.mapping_dict = nn.ModuleDict({
            str(tgt_lang_tok): nn.Linear(feat_channel, embed_dim)
            for tgt_lang_tok in tgt_lang_toks
        })

    def forward(self, feat, tgt_lang_toks, **kwargs):
        densed_feat = torch.zeros(feat.shape[0], feat.shape[1], self.embed_dim).type_as(feat).to(feat.device)

        for tgt_lang_tok in self.tgt_lang_toks:
            mapping_net = self.mapping_dict[str(tgt_lang_tok)]
            is_matched = tgt_lang_toks == tgt_lang_tok
            densed_feat[:, is_matched, :] = mapping_net(feat[:, is_matched, :])
        
        return densed_feat

class LVPGLayer(nn.Module):
    """
    Fusion layer from the paper of EMNLP2022:
    LVP-M3: Language-aware Visual Prompt for Multilingual Multimodal Machine Translation

    """
    def __init__(self, feat_channel, embed_dim, embed_tokens, dropout, **kwargs) -> None:
        super().__init__()
        self.feat_channel = feat_channel
        self.embed_dim = embed_dim

        self.embed_layer_norm = LayerNorm(self.embed_dim)
        self.feat_layer_norm = LayerNorm(self.feat_channel)
        self.embed_tokens = embed_tokens

        self.controller_net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(embed_dim, feat_channel * embed_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )
    
    def forward(self, feat, tgt_lang_toks, **kwargs):
        densed_feat = torch.zeros(feat.shape[0], feat.shape[1], self.embed_dim).type_as(feat).to(feat.device)

        feat = self.feat_layer_norm(feat)

        for tgt_lang_tok in torch.unique(tgt_lang_toks):
            tgt_lang_emb = self.embed_layer_norm(self.embed_tokens(tgt_lang_tok))
            # nn.function.linear takes (out_feat,in_feat)-like weights as the 2nd arg
            mapping_net = self.controller_net(tgt_lang_emb).view(self.embed_dim, self.feat_channel)

            is_matched = tgt_lang_toks == tgt_lang_tok
            densed_feat[:, is_matched, :] = nn.functional.linear(feat[:, is_matched, :], mapping_net, None)

        return densed_feat

PROMPT_GENERATION_MODULES={
    'static': StaticProjectionLayer,
    'separate': SeparateProjectionLayer,
    'lvpg': LVPGLayer
}

@dataclass
class AttentiveFusionPromptConfig(MultimodalTransformerConfig):
    prompt_generation: ChoiceEnum(PROMPT_GENERATION_MODULES.keys()) = field(
        default='lvpg',
        metadata={}
    )

class AttentiveFusionDecoder(MultimodalTransformerDecoder):
    def __init__(
        self,
        cfg: AttentiveFusionPromptConfig,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None
    ):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)

        self.prompt_generator = PROMPT_GENERATION_MODULES[cfg.prompt_generation](
            feat_channel=self.feat_channel,
            embed_dim=self.embed_dim,
            embed_tokens=self.embed_tokens,
            tgt_lang_toks=getattr(cfg, "tgt_lang_toks", None),
            dropout=cfg.dropout,
        )

        self.lvp_encoder_self_attn = self.build_lvp_self_attention(cfg)
        self.lvp_encoder_self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.lvp_feat_self_attn = self.build_lvp_self_attention(cfg)
        self.lvp_feat_self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.lvp_encoder_feat_attn = self.build_lvp_selective_attention(cfg)
        self.lvp_encoder_feat_attn_layer_norm = LayerNorm(self.embed_dim)
    
    def build_lvp_self_attention(self, cfg) -> MaskedMultiheadAttention:
        return MaskedMultiheadAttention(
            self.embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=cfg.quant_noise.pq,
            qn_block_size=cfg.quant_noise.pq_block_size,
            xformers_att_config=cfg.decoder.xformers_att_config,
        )

    def build_lvp_selective_attention(self, cfg) -> MaskedMultiheadAttention:
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

    def extract_multimodal_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        feat = encoder_out['feat'][0]
        feat_padding_mask = encoder_out['feat_padding_mask'][0]
        tgt_lang_toks = encoder_out.get("tgt_lang_tok", [None])[0]
        visual_pompt = self.prompt_generator(feat, tgt_lang_toks=tgt_lang_toks)

        enc_out = encoder_out['encoder_out'][0]
        enc_padding_mask = encoder_out['encoder_padding_mask'][0]

        # self-attention over language
        residual = enc_out
        enc_out = self.lvp_encoder_self_attn_layer_norm(enc_out)
        enc_out, _ = self.lvp_encoder_self_attn(
            query=enc_out,
            key=enc_out,
            value=enc_out,
            key_padding_mask=enc_padding_mask,
            need_weights=False,
            attn_mask=None,
        )
        enc_out = self.dropout_module(enc_out)
        enc_out = self.residual_connection(enc_out, residual)

        # self-attention over feature
        residual = visual_pompt
        visual_pompt = self.lvp_feat_self_attn_layer_norm(visual_pompt)
        visual_pompt, _ = self.lvp_feat_self_attn(
            query = visual_pompt,
            key = visual_pompt,
            value = visual_pompt,
            key_padding_mask = feat_padding_mask,
            need_weights=False,
            attn_mask=None,
        )
        visual_pompt = self.dropout_module(visual_pompt)
        visual_pompt = self.residual_connection(visual_pompt, residual)

        # co-attention
        residual = enc_out
        enc_out = self.lvp_encoder_feat_attn_layer_norm(enc_out)
        enc_out, _ = self.lvp_encoder_feat_attn(
            query = enc_out,
            key = visual_pompt,
            value = visual_pompt,
            key_padding_mask = feat_padding_mask,
            incremental_state = incremental_state,
            static_kv = True,
        )
        enc_out = self.dropout_module(enc_out)
        enc_out = self.residual_connection(enc_out, residual)

        # usual transformer decoder over updated encoder states
        encoder_out['encoder_out'] = [enc_out]        

        return self.extract_features_scriptable(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

@register_model("attentive_fusion")
class AttentiveFusionTransformerModel(MultimodalTransformerModel):
    dataclass_cls = AttentiveFusionPromptConfig
    encoder_cls = MultimodalTransformerEncoder
    decoder_cls = AttentiveFusionDecoder

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg = None,
        args = None
    ):
        # model got a single tgt_lang_toks during the inference step
        # while the model may have 1+ tgt_lang_toks during the training step.
        strict = False
        return super().load_state_dict(state_dict, strict, model_cfg, args)

@register_model_architecture("attentive_fusion", "attentive_fusion_tiny")
def attentive_fusion_tiny(args):
    return transformer_tiny_hirasawa2023(args)

@register_model_architecture("attentive_fusion", "attentive_fusion_small")
def attentive_fusion_small(args):
    return transformer_small_hirasawa2023(args)
