from typing import List, Dict, Optional

import torch
from torch import Tensor

from fairseq.modules import TransformerDecoderLayer
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase

from .utils import MaskedMultiheadAttention, normalize

class AttentiveMultimodalTransformerDecoderLayerBase(TransformerDecoderLayer):
    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        TransformerDecoderLayerBase.__init__(self, cfg, no_encoder_attn, add_bias_kv, add_zero_attn)

        # non-supported options
        assert self.cross_self_attention == False
        assert self.w_resid is None
        assert self.c_attn is None
        assert self.attn_ln is None

        # encoder_attn is mandatory
        assert self.encoder_attn is not None

    def build_feat_attention(self, embed_dim, cfg):
        return MaskedMultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            keep_query=True,
        )

class SerialAttentiveMultimodalTransformerDecoderLayer(AttentiveMultimodalTransformerDecoderLayerBase):
    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(cfg, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.feat_attn = self.build_feat_attention(self.embed_dim, cfg)

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

        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # cross-attention
        ## on modality #1 (text)
        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=need_attn or (not self.training and self.need_attn),
            need_head_weights=need_head_weights,
        )

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        ## on modality #2 (feat)
        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        
        x, _ = self.feat_attn(
            query=x,
            key=feat,
            value=feat,
            key_padding_mask=feat_padding_mask,
            static_kv=True,
            need_weights=False,
        )

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        
        # feed-forward
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

class ParallelAttentiveMultimodalTransformerDecoderLayer(AttentiveMultimodalTransformerDecoderLayerBase):
    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(cfg, no_encoder_attn, add_bias_kv, add_zero_attn)

        self.feat_attn = self.build_feat_attention(self.embed_dim, cfg)

        modality_weights = normalize(torch.Tensor(cfg.modality_weights))
        self.register_buffer('modality_weights', modality_weights)

        assert (cfg.dropnet is None) or (0 <= cfg.dropnet and cfg.dropnet <= 1)
        self.dropnet = cfg.dropnet / 2 if cfg.dropnet else None

    def get_weight_matrix(self, bs, feat_available_mask):
        # 1, n_modality, time, channel
        weight_matrix = self.modality_weights.reshape(1, 2).repeat(bs, 1)

        # apply dropnet only when training
        if self.training:
            rand_matrix = torch.rand(bs)

            weight_matrix[rand_matrix < self.dropnet, 0] = 1.
            weight_matrix[rand_matrix < self.dropnet, 1] = 0.
            weight_matrix[rand_matrix > 1 - self.dropnet, 0] = 0.
            weight_matrix[rand_matrix > 1 - self.dropnet, 1] = 1.

        # use text state if the feat is unavailable
        weight_matrix[~feat_available_mask, 0] = 1.
        weight_matrix[~feat_available_mask, 1] = 0.
        
        return weight_matrix.reshape(-1, 2, 1, 1) # bs, n_modality, time, channel

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

        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # cross-attention
        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        ## on modality #1 (text)
        x_1, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=need_attn or (not self.training and self.need_attn),
            need_head_weights=need_head_weights,
        )  # time(q), bs, channel

        ## on modality #2 (feat)
        feat_available_mask = ~feat_padding_mask.all(dim=-1) # bsz

        valid_x_2, _ = self.feat_attn(
            query=x[:, feat_available_mask, :],
            key=feat[:, feat_available_mask, :],
            value=feat[:, feat_available_mask, :],
            key_padding_mask=feat_padding_mask[feat_available_mask, :],
            static_kv=True,
            need_weights=False,
        )
        x_2 = torch.zeros_like(x_1).to(x_1.device)
        x_2[:, feat_available_mask, :] = valid_x_2 # time(q), bs, channel

        weight_matrix = self.get_weight_matrix(x.shape[1], feat_available_mask)
        x = (weight_matrix * torch.stack([x_1, x_2], dim=1).transpose(0, 2)).sum(dim=1).transpose(0, 1)

        # x = x_1 * self.modality_weights[0] + x_2 * self.modality_weights[1]

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

class FlatAttentiveMultimodalTransformerDecoderLayer(AttentiveMultimodalTransformerDecoderLayerBase):
    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(cfg, no_encoder_attn, add_bias_kv, add_zero_attn)

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

        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # cross-attention
        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        encoder_out = torch.concat([feat, encoder_out], dim=0)
        encoder_padding_mask = torch.concat([feat_padding_mask, encoder_padding_mask], dim=-1)

        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=need_attn or (not self.training and self.need_attn),
            need_head_weights=need_head_weights,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

class HierarchicalAttentiveMultimodalTransformerDecoderLayer(AttentiveMultimodalTransformerDecoderLayerBase):
    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(cfg, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.feat_attn = self.build_feat_attention(self.embed_dim, cfg)
        self.hier_attn = self.build_feat_attention(self.embed_dim, cfg)

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

        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # cross-attention
        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        ## on modality #1 (text)
        x_1, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=need_attn or (not self.training and self.need_attn),
            need_head_weights=need_head_weights,
        )
        x_1 = self.dropout_module(x_1)

        ## on modality #2 (feat)
        x_2, _ = self.feat_attn(
            query=x,
            key=feat,
            value=feat,
            key_padding_mask=feat_padding_mask,
            static_kv=True,
            need_weights=False,
        )
        x_2 = self.dropout_module(x_2)

        ## hier-attention on both modalities
        t_x, bsz, embed_dim = x.shape
        x_modality = torch.stack([x_1, x_2]).reshape(2, t_x * bsz, embed_dim)
        modality_padding_mask = torch.stack([
            encoder_padding_mask.all(dim=-1),
            feat_padding_mask.all(dim=-1),
        ], dim=0).unsqueeze(1).repeat(1, t_x, 1).reshape(2, t_x * bsz).transpose(0, 1)

        x = x.reshape(1, t_x * bsz, embed_dim)
        x, _ = self.hier_attn(
            query=x,
            key=x_modality,
            value=x_modality,
            key_padding_mask=modality_padding_mask,
            static_kv=True,
            need_weights=False,
        )
        x = x.reshape(t_x, bsz, embed_dim)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None
