import logging
from argparse import Namespace
from typing import Dict, List, Optional
from omegaconf import II, DictConfig
from dataclasses import dataclass

import torch
from torch import Tensor

from fairseq.dataclass.utils import gen_parser_from_dataclass

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import base_architecture

from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.transformer.transformer_base import TransformerModelBase
from fairseq.models.transformer.transformer_encoder import TransformerEncoderBase
from fairseq.models.transformer.transformer_decoder import TransformerDecoderBase

from .transformer import transformer_tiny_wu2021
from .utils import unused_and_uninit_parameters

logger = logging.getLogger(__name__)

@dataclass
class MultimodalTransformerConfig(TransformerConfig):
    feat_channel: int = II("task.feat_channel")
    feat_num: int = II("task.feat_num")

    finetune_from_model: Optional[str] = II("checkpoint.finetune_from_model")

class MultimodalTransformerEncoder(TransformerEncoderBase):
    """
    A Transformer encoder that accepts and passes through multimodal features.
    """
    def __init__(
        self,
        cfg: MultimodalTransformerConfig,
        dictionary,
        embed_tokens,
        return_fc=False
    ):
        super().__init__(cfg, dictionary, embed_tokens, return_fc)

        self.feat_channel = cfg.feat_channel
        self.feat_num = cfg.feat_num

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
            tgt_lang_tok=tgt_lang_tok
        )

        return enc_out

    def forward_passthrough_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[Tensor] = None,
        feat: Optional[Tensor] = None,
        feat_padding_mask: Optional[Tensor] = None,
        tgt_lang_tok: Optional[Tensor] = None,
    ):
        enc_out = self.forward_multimodal_scriptable(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=token_embeddings,
            feat=feat, 
            feat_padding_mask=feat_padding_mask,
            tgt_lang_tok=tgt_lang_tok
        )

        # avoid to be modified by in-place functions
        enc_out['feat'] = [feat.clone()]
        enc_out['feat_padding_mask'] = [feat_padding_mask.clone()]
        if tgt_lang_tok is not None:
            enc_out['tgt_lang_tok'] = [tgt_lang_tok]

        return enc_out

    def forward_multimodal_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[Tensor] = None,
        feat: Tensor = None,
        feat_padding_mask: Tensor = None,
        tgt_lang_tok: Tensor = None,
    ):
        """
        Override this function to implement multimodal process 
        to avoid use super().forward_scriptable.
        """
        return self.forward_scriptable(
            src_tokens,
            src_lengths,
            return_all_hiddens,
            token_embeddings
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        ordered_encoder_out = super().reorder_encoder_out(
            encoder_out=encoder_out,
            new_order=new_order
        )

        if len(encoder_out["feat"]) == 0:
            new_feat = []
        else:
            new_feat = [encoder_out["feat"][0].index_select(1, new_order)]

        if len(encoder_out["feat_padding_mask"]) == 0:
            new_feat_padding_mask = []
        else:
            new_feat_padding_mask = [
                encoder_out["feat_padding_mask"][0].index_select(0, new_order)
            ]

        ordered_encoder_out.update({
            "feat": new_feat,   # time, bsz, channel
            "feat_padding_mask": new_feat_padding_mask, # bsz, time
        })

        if ("tgt_lang_tok" in encoder_out):
            if len(encoder_out["tgt_lang_tok"]) == 0:
                new_tgt_lang_tok = []
            else:
                new_tgt_lang_tok = [
                    encoder_out["tgt_lang_tok"][0].index_select(0, new_order)
                ]

            ordered_encoder_out.update({
                "tgt_lang_tok": new_tgt_lang_tok
            })

        return ordered_encoder_out

class MultimodalTransformerDecoder(TransformerDecoderBase):
    """
    A Transformer-based decoder.
    Override `extract_multimodal_features_scriptable` to implementation the multimodal process.
    """
    def __init__(
        self,
        cfg: MultimodalTransformerConfig,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None
    ):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)

        self.feat_channel = cfg.feat_channel
        self.feat_num = cfg.feat_num

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

        # passthrough encoder outputs for computing loss
        for key in ['reconst_feat']:
            if key in encoder_out:
                extra[key] = encoder_out[key]

        return x, extra

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
        Override this function to implement multimodal process 
        to avoid use super().extract_features_scriptable.
        """
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

@register_model("multimodal_transformer")
class MultimodalTransformerModel(TransformerModelBase):
    """
    Base class for all transformer-based multimodal machine translation models
    """
    dataclass_cls = MultimodalTransformerConfig
    encoder_cls = MultimodalTransformerEncoder
    decoder_cls = MultimodalTransformerDecoder

    def __init__(self, args, encoder, decoder):
        cfg = self.dataclass_cls.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args
    
    @classmethod
    def add_args(cls, parser):
        gen_parser_from_dataclass(
            parser, cls.dataclass_cls(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        # copied from the base class #

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )

        # copied from the base class #

        cfg = cls.dataclass_cls.from_namespace(args)
        return super().build_model(cfg, task)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            cls.dataclass_cls.from_namespace(args), dictionary, embed_dim, path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return cls.encoder_cls(
            cls.dataclass_cls.from_namespace(args), src_dict, embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return cls.decoder_cls(
            cls.dataclass_cls.from_namespace(args), tgt_dict, embed_tokens
        )

    def load_state_dict(self, state_dict, strict=True, model_cfg: Optional[DictConfig] = None, args: Optional[Namespace] = None):
        '''
        approve to load a checkpoint of a different architecture.
        '''
        strict = strict and (model_cfg.finetune_from_model is None)
        
        if not strict:
            logger.info('load a checkpoint unstrictly')
            unused, uninit = unused_and_uninit_parameters(self, state_dict)
            logger.info('Unused parameters: ' + ','.join(unused))
            logger.info('Uninit parameters: ' + ','.join(uninit))

        return super().load_state_dict(state_dict, strict, model_cfg, args)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        feat: Optional[Tensor] = None,
        feat_padding_mask: Optional[Tensor] = None,
        tgt_lang_tok: Optional[Tensor] = None,
    ):
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            feat=feat,
            feat_padding_mask=feat_padding_mask,
            tgt_lang_tok=tgt_lang_tok,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

@register_model_architecture("multimodal_transformer", "multimodal_transformer_tiny_wu2021")
def multimodal_transformer_tiny_wu2021(args):
    transformer_tiny_wu2021(args)
