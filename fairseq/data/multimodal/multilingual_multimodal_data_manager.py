import logging

# from fairseq.tasks.multimodal_translation import MultimodalTranslationConfig
from fairseq.data.multilingual.multilingual_data_manager import (
    MultilingualDatasetManager,
    TGT_DICT_NAME
)

from .multimodal_language_pair_dataset import (
    MultimodalLanguagePairDataset,
    load_feat_data
)

logger = logging.getLogger(__name__)

class MultilingualMultimodalDatasetManager(MultilingualDatasetManager):
    def __init__(self, 
        args, lang_pairs=None, langs=None, dicts=None, sampling_method=None, 
        multilingual_dataset_manager=None
    ):
        if multilingual_dataset_manager is None:
            super().__init__(args, lang_pairs, langs, dicts, sampling_method)
        else:
            self.__dict__.update(multilingual_dataset_manager.__dict__)
        
        self.tgt_lang_toks = [
            self.get_decoder_langtok(tgt_lang, args.lang_tok_style) 
            for tgt_lang in self.tgt_langs if self.has_target_dictionary(tgt_lang)
        ]

    @classmethod
    def setup_data_manager(cls, args, lang_pairs, langs, dicts, sampling_method):
        return MultilingualMultimodalDatasetManager(
            args=args, lang_pairs=lang_pairs, langs=langs, dicts=dicts, sampling_method=sampling_method
        )

    @classmethod
    def from_base(cls, args, multilingual_dataset_manager):
        return MultilingualMultimodalDatasetManager(
            args,
            multilingual_dataset_manager=multilingual_dataset_manager
        )

    def has_target_dictionary(self, lang):
        if self.args.target_dict is not None:
            return TGT_DICT_NAME in self.dicts
        else:
            return lang in self.dicts

    def load_a_dataset(
        self,
        split,
        data_path,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        prepend_bos=False,
        langpairs_sharing_datasets=None,
        data_category=None,
        **extra_kwargs,
    ):
        langpair_ds = super().load_a_dataset(
            split,
            data_path,
            src,
            src_dict,
            tgt,
            tgt_dict,
            combine,
            prepend_bos,
            langpairs_sharing_datasets,
            data_category,
            **extra_kwargs,
        )

        feat_data, feat_padding_mask_data, feat_indices_data = load_feat_data(
            data_path,
            split,
            src,
            tgt,
            self.args.feat_infix_or_path,
            self.args.feat_padding_mask_infix_or_path,
            self.args.feat_indices_infix_or_path
        )

        assert len(langpair_ds) == feat_data.shape[0]

        ds = MultimodalLanguagePairDataset(
            langpair_dataset=langpair_ds,
            feat_channel=self.args.feat_channel,
            feat_num=self.args.feat_num,
            feat=feat_data,
            feat_padding_mask=feat_padding_mask_data,
            feat_indices=feat_indices_data,
            enable_tgt_lang_tok=self.args.enable_tgt_lang_tok,
            max_items=extra_kwargs.get('max_items', None),
        )

        return ds
