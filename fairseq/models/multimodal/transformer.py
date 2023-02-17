from fairseq.models import (
    register_model_architecture,
)
from fairseq.models.transformer.transformer_legacy import base_architecture

@register_model_architecture("transformer", "transformer_tiny_wu2021")
def transformer_tiny_wu2021(args):
    """
    Tiny Transformer model from
    "Good for Misconceived Reasons: An Empirical Revisiting on the Need for Visual Context in Multimodal Machine Translation"
    Zhiyong Wu, Lingpeng Kong, Wei Bi, Xiang Li, Ben Kao. ACL 2021.
    
    Use following arguments to reproduce the original model:
        --warmup-updates 2000
        --lr 0.005
        --max-tokens 4096
        --max-update 8000
        --patience 10
    """
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 128)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 256)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

    args.dropout = getattr(args, "dropout", 0.3)

    return base_architecture(args)

@register_model_architecture("transformer", "transformer_tiny_fan2020")
def transformer_tiny_fan2020(args):
    """
    The tiny version of Transformer for multilingual machine translation proposed by
    "Beyond English-Centric Multilingual Machine Translation"
    https://www.jmlr.org/papers/v22/20-1307.html
    """

    # shared all embeddings (enc, dec, out)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    # LayerDrop
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.05)

    # PreNorm
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)

    return transformer_tiny_wu2021(args)

@register_model_architecture("transformer", "transformer_tiny_hirasawa2023")
def transformer_tiny_hirasawa2023(args):
    """
    Variant of Wu+, 2021 model.
     - layernorm embeddings prior to passing them into encoder/decoder layers
     
    """

    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.05)

    return transformer_tiny_wu2021(args)

@register_model_architecture("transformer", "transformer_tiny_hirasawa2023_dedicated_vocab")
def transformer_tiny_hirasawa2023_dedicated_vocab(args):
    """
    Variant of Wu+, 2021 model.
     - layernorm embeddings prior to passing them into encoder/decoder layers
     
    """

    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.05)

    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", True)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 128)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 256)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

    args.dropout = getattr(args, "dropout", 0.3)

    return base_architecture(args)

@register_model_architecture("transformer", "transformer_base_hirasawa2023")
def transformer_base_hirasawa2023(args):
    """
    Variant of Wu+, 2021 model.
     - layernorm embeddings prior to passing them into encoder/decoder layers
     
    """

    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.05)

    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    
    args.dropout = getattr(args, "dropout", 0.3)

    return base_architecture(args)

@register_model_architecture("transformer", "transformer_small_hirasawa2023")
def transformer_small_hirasawa2023(args):
    """
    Variant of Wu+, 2021 model.
     - layernorm embeddings prior to passing them into encoder/decoder layers
     
    """

    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.05)

    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

    args.dropout = getattr(args, "dropout", 0.3)

    return base_architecture(args)
