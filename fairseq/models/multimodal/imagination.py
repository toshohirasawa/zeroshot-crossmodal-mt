from typing import Optional, Dict, List

import torch
from torch import Tensor

from fairseq.models import (
    register_model,
    register_model_architecture,
)

from .utils import (
    mask_pad_elements,
)
from .encoder import ProjectionalMultimodalTransformerEncoder
from .transformer import (
    transformer_tiny_wu2021,
    transformer_tiny_fan2020,
    transformer_tiny_hirasawa2023,
    transformer_small_hirasawa2023,
    transformer_base_hirasawa2023,
)
from .multimodal_transformer import (
    MultimodalTransformerModel,
    MultimodalTransformerDecoder,
)

from .imagination_config import (
    ImaginationMultimodalTransformerConfig,
    MULTIMODAL_DECODER_MODELS
)
from .imagination_module import build_predictor

torch.autograd.set_detect_anomaly(True)

class ImaginationMultimodalTransformerEncoder(ProjectionalMultimodalTransformerEncoder):
    """
    Transformer-based Imagination model
    """

    def __init__(
        self,
        cfg: ImaginationMultimodalTransformerConfig,
        dictionary,
        embed_tokens,
        return_fc=False
    ):
        super().__init__(cfg, dictionary, embed_tokens, return_fc)

        self.feat_predictor = build_predictor(cfg.predictor.model, cfg)
    
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
        enc_out = self.forward_passthrough_scriptable(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=token_embeddings,
            feat=feat, 
            feat_padding_mask=feat_padding_mask,
            tgt_lang_tok=tgt_lang_tok,
        )

        encoder_padding_mask = enc_out['encoder_padding_mask'][0]
        masked_enc_out = mask_pad_elements(
            enc_out['encoder_out'][0],
            encoder_padding_mask,
        )
        feat_pred = self.feat_predictor(masked_enc_out, encoder_padding_mask)
        enc_out['feat_pred'] = [feat_pred]

        if self.cfg.replace_all_by_pred:
            feat = feat_pred
            enc_out['feat_padding_mask'][0][:, :] = False
            
        elif self.cfg.replace_pad_by_pred:
            new_feat = torch.zeros_like(feat).type_as(feat).to(feat.device)
            
            feat_padding_mask = feat_padding_mask.transpose(0, 1)
            new_feat[~feat_padding_mask] = feat[~feat_padding_mask]
            new_feat[feat_padding_mask] = feat_pred[feat_padding_mask]

            feat = new_feat
            enc_out['feat_padding_mask'][0][:, :] = False

        densed_feat = self.projection_layer_norm(feat)
        densed_feat = self.projection(densed_feat)

        enc_out['feat'] = [feat]
        enc_out['densed_feat'] = [densed_feat]

        return enc_out

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        ordered_encoder_out = super().reorder_encoder_out(
            encoder_out=encoder_out,
            new_order=new_order
        )

        if len(encoder_out["feat_pred"]) == 0:
            new_feat_pred = []
        else:
            new_feat_pred = [encoder_out["feat_pred"][0].index_select(1, new_order)]


        ordered_encoder_out.update({
            "feat_pred": new_feat_pred,   # time, bsz, channel
        })

        return ordered_encoder_out

class ImaginationMultimodalTransformerDecoder(MultimodalTransformerDecoder):
    def __init__(
        self,
        cfg: ImaginationMultimodalTransformerConfig,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None
    ):
        # transfer all attributes of the base decoder
        # to enable training the model from a pre-trained transformer model.
        base_decoder = MULTIMODAL_DECODER_MODELS[cfg.multimodal_decoder_model](
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )
        self.__dict__.update(base_decoder.__dict__)
        self.extract_multimodal_features_scriptable = base_decoder.extract_multimodal_features_scriptable

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        x, extra = self.extract_multimodal_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

        extra['feat_pred'] = encoder_out['feat_pred'][0]

        return x, extra

@register_model("imagination")
class ImaginationMultimodalTransformerModel(MultimodalTransformerModel):
    dataclass_cls = ImaginationMultimodalTransformerConfig
    encoder_cls = ImaginationMultimodalTransformerEncoder
    decoder_cls = ImaginationMultimodalTransformerDecoder

    @classmethod
    def build_model(cls, args, task):
        args.predictor_output_dim = getattr(args, "predictor_output_dim", args.feat_channel)
        args.predictor_output_num = getattr(args, "predictor_output_num", args.feat_num)
        args.predictor_input_dim = getattr(args, "predictor_input_dim", args.encoder_embed_dim)
        
        return super().build_model(args, task)    

@register_model_architecture("imagination", "imagination_tiny")
def imagination_tiny(args):

    transformer_tiny_hirasawa2023(args)
    
    args.predictor_embed_path = getattr(args, "predictor_embed_path", None)
    args.predictor_embed_dim = getattr(args, "predictor_embed_dim", args.encoder_embed_dim)
    args.predictor_ffn_embed_dim = getattr(
        args, "predictor_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.predictor_layers = getattr(args, "predictor_layers", 1)
    args.predictor_attention_heads = getattr(args, "predictor_attention_heads", 4)
    args.predictor_normalize_before = getattr(args, "predictor_normalize_before", True)
    args.predictor_learned_pos = getattr(args, "predictor_learned_pos", False)
    args.predictor_layers_to_keep = getattr(args, "predictor_layers_to_keep", None)
    args.predictor_layerdrop = getattr(args, "predictor_layerdrop", 0)

@register_model_architecture("imagination", "imagination_small")
def imagination_small(args):

    transformer_small_hirasawa2023(args)
    
    args.predictor_embed_path = getattr(args, "predictor_embed_path", None)
    args.predictor_embed_dim = getattr(args, "predictor_embed_dim", args.encoder_embed_dim)
    args.predictor_ffn_embed_dim = getattr(
        args, "predictor_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.predictor_layers = getattr(args, "predictor_layers", 1)
    args.predictor_attention_heads = getattr(args, "predictor_attention_heads", 4)
    args.predictor_normalize_before = getattr(args, "predictor_normalize_before", True)
    args.predictor_learned_pos = getattr(args, "predictor_learned_pos", False)
    args.predictor_layers_to_keep = getattr(args, "predictor_layers_to_keep", None)
    args.predictor_layerdrop = getattr(args, "predictor_layerdrop", 0)
