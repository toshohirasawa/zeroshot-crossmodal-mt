import logging

from fairseq.data import (
    MultimodalLanguagePairDataset,
    ListDataset,
)
from fairseq.data.multimodal.multilingual_multimodal_data_manager import MultilingualMultimodalDatasetManager

from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)

@register_task("multilingual_multimodal_translation")
class MultimodalMultilingualTranslationTask(TranslationMultiSimpleEpochTask):
    """
    Translate from one (source) language to another (target) language along with additional modalities.
    Args:
        langs (List[str]): a list of languages that are being supported
        dicts (Dict[str, fairseq.data.Dictionary]): mapping from supported languages to their dictionaries
        training (bool): whether the task should be configured for training or not
    .. note::
        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    The translation task provides the following additional command-line
    arguments:
    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        TranslationMultiSimpleEpochTask.add_args(parser)

        #TODO: consider to use a dataclass
        parser.add_argument("--feat-channel", type=int, default=2048, help="the channel of features")
        parser.add_argument("--feat-num", type=int, default=36, help="the number of features")

        parser.add_argument("--feat-infix-or-path", type=str, default="faster-rcnn",
                            help="a name or path of the feature file (npy format)")
        parser.add_argument("--feat-padding-mask-infix-or-path", type=str, default=None,
                            help="a name or path of the feature padding file (raw text format)")
        parser.add_argument("--feat-indices-infix-or-path", type=str, default=None,
                            help="a name or path of the feature padding file (raw text format)")
        parser.add_argument("--enable-tgt-lang-tok", action="store_true",
                            help="wether pass tgt_lang_tok id to the model")
        
    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)


        self.data_manager = MultilingualMultimodalDatasetManager.from_base(args, self.data_manager)

        # id of target-language tokens in embedding
        args.tgt_lang_toks = self.data_manager.tgt_lang_toks

    @classmethod
    def setup_task(cls, args, **kwargs):
        langs, dicts, training = MultilingualMultimodalDatasetManager.prepare(
            cls.load_dictionary, args, **kwargs
        )
        return cls(args, langs, dicts, training)
    
    def build_model(self, cfg, from_checkpoint=False):
        cfg.tgt_lang_toks = self.args.tgt_lang_toks
        return super().build_model(cfg, from_checkpoint)

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the multilingual_translation task is not supported"
            )

        src_data = ListDataset(src_tokens, src_lengths)
        dataset = MultimodalLanguagePairDataset(src_data, src_lengths, self.source_dictionary, feat= self.args.feat_infix_or_path)
        src_langtok_spec, tgt_langtok_spec = self.args.langtoks["main"]
        if self.args.lang_tok_replacing_bos_eos:
            dataset = self.data_manager.alter_dataset_langtok(
                dataset,
                src_eos=self.source_dictionary.eos(),
                src_lang=self.args.source_lang,
                tgt_eos=self.target_dictionary.eos(),
                tgt_lang=self.args.target_lang,
                src_langtok_spec=src_langtok_spec,
                tgt_langtok_spec=tgt_langtok_spec,
            )
        else:
            dataset.src = self.data_manager.src_dataset_tranform_func(
                self.args.source_lang,
                self.args.target_lang,
                dataset=dataset.src,
                spec=src_langtok_spec,
            )
        return dataset