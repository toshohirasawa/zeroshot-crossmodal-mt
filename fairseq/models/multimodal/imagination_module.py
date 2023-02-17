from typing import List, Optional

import torch
from torch import nn, Tensor

from fairseq import utils
from fairseq.data.dictionary import Dictionary
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    MultiheadAttention,
)

from .imagination_config import IMAGINATION_POOLINGS

class LinearSequentialFeaturePredictor(nn.Module):
    """
    A naive projection-based Imagination model for sequential feature (e.g. local or RoI feature)
    """
    def __init__(self, embed_dim, feat_channel, feat_num, pooling_type) -> None:
        super().__init__()

        self.feat_channel = feat_channel
        self.feat_num = feat_num
        self.pooling_type = pooling_type

        self.pooling = IMAGINATION_POOLINGS[pooling_type]
        self.layer_norm = LayerNorm(embed_dim)
        self.predict = nn.Linear(embed_dim, feat_channel * feat_num)

    def forward(self, encoder_out, encoder_padding_mask) -> List[Tensor]:
        _, bsz, _ = encoder_out.shape
        pooled_enc_out = self.pooling(encoder_out, encoder_padding_mask)

        pooled_enc_out = self.layer_norm(pooled_enc_out)
        predicted_feat = self.predict(pooled_enc_out).reshape(self.feat_num, bsz, self.feat_channel)

        return predicted_feat

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.predictor.input_dim, cfg.feat_channel, cfg.feat_num, cfg.predictor.pooling_type)

class TransformerFeaturePredictorLayer(nn.Module):
    def __init__(self,
        embed_dim: int,
        attention_heads: int,
        encoder_embed_dim: int,
        ffn_embed_dim: int,
        dropout: float,
        activation_fn: str,
    ) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        
        self.self_attn = MultiheadAttention(
            embed_dim = self.embed_dim,
            num_heads=attention_heads,
            dropout=dropout,
            self_attention=True,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.encoder_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=attention_heads,
            kdim=encoder_embed_dim,
            vdim=encoder_embed_dim,
            dropout=dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(embed_dim)
    
    def forward(
        self, 
        x, 
        encoder_out: Optional[Tensor],
        encoder_padding_mask: Optional[Tensor],
    ):
        x = self.forward_self_attn(x)
        x = self.forward_encoder_attn(x, encoder_out, encoder_padding_mask)
        x = self.forward_ff(x)

        return x
    
    def forward_self_attn(self, x):
        residual = x

        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        return x

    def forward_encoder_attn(
        self, 
        x, 
        encoder_out: Optional[Tensor],
        encoder_padding_mask: Optional[Tensor],
    ):
        residual = x

        x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            static_kv=True,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        return x

    def forward_ff(self, x):
        residual = x

        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        return x

    def residual_connection(self, x, residual):
        return residual + x

class TransformerFeaturePredictor(nn.Module):
    def __init__(
        self,
        layers: int,
        embed_dim: int,
        attention_heads: int,
        ffn_embed_dim: int,
        activation_fn: str,
        encoder_embed_dim: int,
        output_embed_dim: int,
        output_num: int,
        dropout: float,
        layerdrop: float,
    ):
        super().__init__()

        dictionary = Dictionary(extra_special_symbols=list(range(output_num)))

        self.embed_dim = embed_dim
        self.output_embed_dim = output_embed_dim
        self.output_num = output_num
        self.layerdrop = layerdrop

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        # embeddings and positions
        n_vocab = len(dictionary)
        self.embed_tokens =  nn.Embedding(n_vocab, embed_dim, dictionary.pad_index)
        self.embed_positions = PositionalEmbedding(
                output_num,
                embed_dim,
                dictionary.pad_index,
            )
        
        self.layernorm_embedding = LayerNorm(embed_dim)

        # layers
        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        self.layers.extend([ 
            TransformerFeaturePredictorLayer(
                embed_dim=embed_dim,
                attention_heads=attention_heads,
                encoder_embed_dim=encoder_embed_dim,
                ffn_embed_dim=ffn_embed_dim,
                dropout=dropout,
                activation_fn=activation_fn
            ) for _ in range(layers)
        ])

        # output
        self.output_layernorm = LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, output_embed_dim, bias=True)

        self.register_buffer("output_tokens", torch.LongTensor([range(n_vocab)[-output_num:]]))
    
    def forward(self, enc, padding_mask):
        _, bsz, _ = enc.shape
        input_tokens = self.output_tokens.repeat(bsz, 1)

        # embed tokens and positions
        x = self.embed_tokens(input_tokens.contiguous()) + self.embed_positions(input_tokens)
        x = self.layernorm_embedding(x)
        x = self.dropout_module(x)

        # bsz, output_num, channel -> time(q), output_num, channel
        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, enc, padding_mask,)

        # output projection
        x = self.output_layernorm(x)
        x = self.output_projection(x)

        return x
    
    @classmethod
    def from_config(cls, cfg):
        return cls(
            layers=cfg.predictor.layers,
            embed_dim=cfg.predictor.embed_dim,
            attention_heads=cfg.predictor.attention_heads,
            ffn_embed_dim=cfg.predictor.ffn_embed_dim,
            activation_fn=cfg.activation_fn,
            encoder_embed_dim=cfg.encoder.embed_dim,
            output_embed_dim=cfg.feat_channel,
            output_num=cfg.feat_num,
            dropout=cfg.dropout,
            layerdrop=cfg.predictor.layerdrop,
        )

PREDICTOR_MODELS = {
    "linear": LinearSequentialFeaturePredictor,
    "transformer": TransformerFeaturePredictor,
}

def build_predictor(name, cfg):
    return PREDICTOR_MODELS[name].from_config(cfg)