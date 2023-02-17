from typing import Optional, Tuple, Dict

import torch
from torch import Tensor

from fairseq.modules import MultiheadAttention

def mask_pad_elements(x: Tensor, padding_mask: Tensor):
    """
    Args:
        x (Tensor): Tensor of shape `(time_x, bsz, channel)`
        padding_mask (BoolTensor): binary Tensor of shape `(bsz, time_x)`
    """
    padding_mask = padding_mask.transpose(0, 1).unsqueeze(-1).type_as(x)
    x = x * (1 - padding_mask)
    return x

def masked_mean_pooling(token_emb: Tensor, token_padding_mask: Tensor):
    token_padding_mask = token_padding_mask.transpose(0, 1).unsqueeze(-1) # seq_len, batch_size, 1

    pooled_emb = torch.sum(token_emb.masked_fill(token_padding_mask, 0.), dim=0, keepdim=True)
    pooled_emb = pooled_emb / (~token_padding_mask).sum(dim=0, keepdim=True)
    return pooled_emb

def masked_max_pooling(token_emb: Tensor, token_padding_mask: Tensor):
    token_padding_mask = token_padding_mask.transpose(0, 1).unsqueeze(-1) # seq_len, batch_size, 1

    pooled_emb, max_indices = torch.max(token_emb.masked_fill(token_padding_mask, -torch.inf), dim=0, keepdim=True)
    return pooled_emb


class MaskedMultiheadAttention(MultiheadAttention):
    """
    MultiHeadAttenion will return an `NaN` gradient for fully-masked keys
    https://github.com/pytorch/pytorch/issues/41508

    This class provides a work-around solution by masked-out fully-masked keys
    prior to passing them to MultiHeadAttention.forward function,
    and insteadlly fill the corresponding positions at the output with 0
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        # TODO: pass in config rather than string.
        # config defined in xformers.components.attention.AttentionConfig
        xformers_att_config: Optional[str] = None,
        xformers_blocksparse_layout: Optional[
            torch.Tensor
        ] = None,  # This should be part of the config
        xformers_blocksparse_blocksize: Optional[
            int
        ] = 16,  # This should be part of the config
        # 
        keep_query = False,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            self_attention,
            encoder_decoder_attention,
            q_noise,
            qn_block_size,
            xformers_att_config,
            xformers_blocksparse_layout,
            xformers_blocksparse_blocksize
        )
        self.keep_query = keep_query
    
    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False  ,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if need_weights:
            raise NotImplementedError("`need_weights` is currently not supported.")

        key_fully_padding_mask = key_padding_mask.all(dim=-1)
        has_key_mask = ~key_fully_padding_mask

        # need not run masked multi-head attention
        if (
            (key_padding_mask is None) or # no padding mask
            (key_fully_padding_mask.sum() == 0) # no example with a fully-padding feature
        ):
            return super().forward(
                query,
                key,
                value,
                key_padding_mask,
                incremental_state,
                need_weights,
                static_kv,
                attn_mask,
                before_softmax,
                need_head_weights
            )
        # return query or empty
        elif (
            has_key_mask.sum() == 0 # all features are fully-padded
        ):
            if self.keep_query:
                return query, None
            else:
                return torch.zeros_like(query).type_as(query).to(query.device), None

        out_state = torch.zeros_like(query).type_as(query).to(query.device)

        valid_query = query[:, has_key_mask, :]
        valid_key = key[:, has_key_mask, :]
        valid_value = value[:, has_key_mask, :]
        valid_key_padding_mask = key_padding_mask[has_key_mask, :]

        valid_out = super().forward(
            valid_query,
            valid_key,
            valid_value,
            valid_key_padding_mask,
            incremental_state,
            need_weights,
            static_kv,
            attn_mask,
            before_softmax,
            need_head_weights
        )

        out_state[:, has_key_mask, :] = valid_out[0]
        
        if self.keep_query:
            out_state[:, ~has_key_mask, :] = query[:, ~has_key_mask, :]

        return out_state, *valid_out[1:]


def unused_and_uninit_parameters(model, state_dict):
    model_param_names = set([p[0] for p in model.named_parameters()])
    state_param_names = set(state_dict.keys())
    common_params = model_param_names & state_param_names

    return list(state_param_names - common_params), list(model_param_names - common_params)

def normalize(v: Tensor):
    assert v.dim() == 1

    vn = torch.norm(v).detach()
    v = v.div(vn.expand_as(v))

    return v