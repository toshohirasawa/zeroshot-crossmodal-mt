from dataclasses import dataclass, field
from typing import Optional
import logging

from fairseq import utils
from fairseq.data.multimodal.multimodal_language_pair_dataset import (
    MultimodalLanguagePairDataset,
    load_feat_data
)
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, TranslationConfig, load_langpair_dataset

logger = logging.getLogger(__name__)

@dataclass
class MultimodalTranslationConfig(TranslationConfig):
    feat_channel: int = field(
        default=2048,
        metadata={"help": "dimension of feature"},
    )
    feat_num: int = field(
        default=36,
        metadata={"help": "Number of multimodal feature"}
    )
    feat_infix_or_path: str = field(
        default="resnet50-avgpool",
        metadata={"help": "a name or path of the feature file (npy format)"}
    )
    feat_padding_mask_infix_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "a name or path of the feature padding file (raw text format)"}
    )
    feat_indices_infix_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "a name or path of the feature padding file (raw text format)"}
    )

@register_task("multimodal_translation", dataclass=MultimodalTranslationConfig)
class MultimodalTranslationTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.
    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language
    .. note::
        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: MultimodalTranslationConfig

    def __init__(self, cfg: MultimodalTranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    @classmethod
    def setup_task(cls, cfg: MultimodalTranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        base_task = TranslationTask.setup_task(cfg, **kwargs)

        return cls(cfg, base_task.src_dict, base_task.tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        langpair_dataset = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )

        feat_dataset, feat_padding_mask_dataset, feat_indices_dataset = load_feat_data(
            data_path, split, src, tgt, 
            self.cfg.feat_infix_or_path, 
            self.cfg.feat_padding_mask_infix_or_path,
            self.cfg.feat_indices_infix_or_path,
        )

        self.datasets[split] = MultimodalLanguagePairDataset(
            langpair_dataset=langpair_dataset,
            feat_num=self.cfg.feat_num,
            feat_channel=self.cfg.feat_channel,
            feat=feat_dataset,
            feat_padding_mask=feat_padding_mask_dataset,
            feat_indices=feat_indices_dataset,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        langpair_dataset = super().build_dataset_for_inference(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )
        feat_dataset, feat_padding_mask_dataset, feat_indices_dataset = load_feat_data(
            None, None, None, None, 
            self.cfg.feat_infix_or_path, 
            self.cfg.feat_padding_mask_infix_or_path,
            self.cfg.feat_indices_infix_or_path,
        )
        return MultimodalLanguagePairDataset(
            langpair_dataset=langpair_dataset,
            feat_num=self.cfg.feat_num,
            feat_channel=self.cfg.feat_channel,
            feat=feat_dataset,
            feat_padding_mask=feat_padding_mask_dataset,
            feat_indices=feat_indices_dataset,
        )
