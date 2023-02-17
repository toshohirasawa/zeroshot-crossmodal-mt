import re
from dataclasses import dataclass, field, fields
from typing import Optional, List


from fairseq.dataclass import ChoiceEnum
from fairseq.utils import safe_getattr, safe_hasattr
from fairseq.models.transformer.transformer_config import (
    EncDecBaseConfig,
    DecoderConfig,
    QuantNoiseConfig
)

from .multimodal_transformer import (
    MultimodalTransformerConfig,
    MultimodalTransformerDecoder,
)
from .concat import ConcatMultimodalTransformerDecoder
from .gated import GatedMultimodalTransformerDecoder
from .gated_attentive import GatedAttentiveMultimodalTransformerDecoder
from .gated_xattn_dense import GatedXattnDenseTransformerDecoder
from .multimodal_projection_layers import AttentiveFusionDecoder, PROMPT_GENERATION_MODULES
from .utils import (
    masked_mean_pooling,
    masked_max_pooling,
)

IMAGINATION_POOLINGS = {
    "mean": masked_mean_pooling,
    "max": masked_max_pooling,
}

MULTIMODAL_DECODER_MODELS = {
    "text-only": MultimodalTransformerDecoder,
    "concat": ConcatMultimodalTransformerDecoder,
    "gated": GatedMultimodalTransformerDecoder,
    "gated-attentive": GatedAttentiveMultimodalTransformerDecoder,
    "gated-xattn-dense": GatedXattnDenseTransformerDecoder,
    "attentive_fusion": AttentiveFusionDecoder,
}

@dataclass
class FeaturePredictorConfig(DecoderConfig):
    """
    Feature predictor configuration
    """
    model: ChoiceEnum(["linear", "transformer"]) = field(
        default="linear",
        metadata={
            "help": "predictor model"
        }
    )

    # linear
    pooling_type: ChoiceEnum(IMAGINATION_POOLINGS.keys()) = field(
        default="mean",
        metadata={"help": "pooling type used in the linear predictor"}
    )

@dataclass
class ImaginationMultimodalTransformerConfig(MultimodalTransformerConfig):
    """
    Imagination configuration
    """

    # feature predicator
    predictor: FeaturePredictorConfig = FeaturePredictorConfig()

    replace_pad_by_pred: Optional[bool] = field(
        default=False,
        metadata={"help": "Replace padded features by predicted features"}
    )
    replace_all_by_pred: Optional[bool] = field(
        default=False,
        metadata={"help": "replace all features by predicted features"}
    )

    # multimodal decoder
    multimodal_decoder_model: ChoiceEnum(MULTIMODAL_DECODER_MODELS.keys()) = field(
        default="text-only",
        metadata={"help": "Model of the decoder"}
    )

    ## attentive_fusion
    prompt_generation: ChoiceEnum(PROMPT_GENERATION_MODULES.keys()) = field(
        default='lvpg',
        metadata={}
    )

    # parser customized
    _NAME_PARSER = r"(decoder|encoder|quant_noise|predictor)_(.*)"

    def __getattr__(self, name, name_parser=_NAME_PARSER):
        match = re.match(name_parser, name)
        if match:
            sub = safe_getattr(self, match[1])
            return safe_getattr(sub, match[2])
        raise AttributeError(f"invalid argument {name}.")

    def __setattr__(self, name, value, name_parser=_NAME_PARSER):
        match = re.match(name_parser, name)
        if match:
            sub = safe_getattr(self, match[1])
            setattr(sub, match[2], value)
        else:
            super().__setattr__(name, value)

    @classmethod
    def from_namespace(cls, args):
        if args is None:
            return None
        if not isinstance(args, cls):
            seen = set()
            config = cls()
            # currently, we can go generically from DC fields to args hierarchically
            # but we can't easily deconstruct a flat namespace to a hierarchical
            # DC. Mostly because we could have a sub-dc called `decoder-foo` that should not
            # go to the sub struct called `decoder`. There are ways to go around this, but let's keep it simple
            # for now.
            for fld in fields(cls):
                # concretelly, the transformer_config know what sub-dc it has, so we go through all the dc fields
                # and if it's one that has a sub-dc, we build that sub-dc with `copy_keys()`
                if fld.name == "decoder":
                    if safe_hasattr(args, "decoder"):
                        #  in some cases, the args we receive is already structured (as DictConfigs), so let's just build the correct DC
                        seen.add("decoder")
                        config.decoder = DecoderConfig(**args.decoder)
                    else:
                        config.decoder = cls._copy_keys(
                            args, DecoderConfig, "decoder", seen
                        )
                elif fld.name == "encoder":
                    # same but for encoder
                    if safe_hasattr(args, "encoder"):
                        seen.add("encoder")
                        config.encoder = EncDecBaseConfig(**args.encoder)
                    else:
                        config.encoder = cls._copy_keys(
                            args, EncDecBaseConfig, "encoder", seen
                        )
                elif fld.name == "quant_noise":
                    # same but for quant_noise
                    if safe_hasattr(args, "quant_noise"):
                        seen.add("quant_noise")
                        config.quant_noise = QuantNoiseConfig(**args.quant_noise)
                    else:
                        config.quant_noise = cls._copy_keys(
                            args, QuantNoiseConfig, "quant_noise", seen
                        )
                # added to take predictor into account
                elif fld.name == "predictor":
                    # same but for quant_noise
                    if safe_hasattr(args, "predictor"):
                        seen.add("predictor")
                        config.predictor = FeaturePredictorConfig(**args.predictor)
                    else:
                        config.predictor = cls._copy_keys(
                            args, FeaturePredictorConfig, "predictor", seen
                        )
                elif safe_hasattr(args, fld.name):
                    # if it's not a structure field, it's just a normal field, copy it over
                    seen.add(fld.name)
                    setattr(config, fld.name, safe_getattr(args, fld.name))
            # we got all the fields defined in the dataclass, but
            # the argparse namespace might have extra args for two reasons:
            #   - we are in a legacy class so all the args are not declared in the dataclass. Ideally once everyone has defined a dataclass for their model, we won't need this
            #   - some places expect args to be there but never define them
            args_dict = (
                args._asdict()
                if safe_hasattr(args, "_asdict")
                else vars(args)
                if safe_hasattr(args, "__dict__")
                else {}
            )  # namedtupled doesn't have __dict__ :-/
            for key, value in args_dict.items():
                if key not in seen:
                    setattr(config, key, value)
            return config
        else:
            return args
