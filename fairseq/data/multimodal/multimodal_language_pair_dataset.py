import os
import logging

from typing import List

import numpy as np
import torch
from fairseq.data import LanguagePairDataset

logger = logging.getLogger(__name__)

def find_data_path(data_path, split, src, tgt, infix_or_path, ext):
    if infix_or_path is None:
        return None
    elif os.path.exists(infix_or_path):
        return infix_or_path
    else:
        for langpair in [f"{src}-{tgt}", f"{tgt}-{src}"]:
            filename = os.path.join(data_path, "{}.{}.{}.{}".format(split, langpair, infix_or_path, ext))
            if os.path.exists(filename):
                return filename
        else:
            logger.info(f'Cannot find data path with ({split}, {src}, {tgt}, {infix_or_path}, {ext})')
            return None

def load_feat_data(data_path, split, src, tgt, 
        feat_infix_or_path, 
        feat_padding_mask_infix_or_path,
        feat_indices_infix_or_path,
    ):
    feat_path = find_data_path(data_path, split, src, tgt, feat_infix_or_path, 'npy')
    feat_padding_path = find_data_path(data_path, split, src, tgt, feat_padding_mask_infix_or_path, 'txt')
    feat_indices_path = find_data_path(data_path, split, src, tgt, feat_indices_infix_or_path, 'txt')

    if feat_path is None:
        logger.error(
            f"Dataset not found: {data_path}, {split}, {src}, {tgt}, {feat_infix_or_path}"
        )
        raise FileNotFoundError(
            "Dataset not found: {} ({})".format(split, data_path)
        )

    feat = torch.from_numpy(np.load(feat_path)).float()

    if feat_padding_path is None:
        feat_padding_mask = None
    else:
        logger.info(f'Loading feature padding from {feat_padding_path}')
        feat_padding_mask = torch.BoolTensor([int(l.strip()) for l in open(feat_padding_path)])

    if feat_indices_path is None:
        feat_indices = None
    else:
        logger.info(f'Loading feature indices from {feat_indices_path}')
        feat_indices = [int(l.strip()) for l in open(feat_indices_path)]

    return feat, feat_padding_mask, feat_indices

class MultimodalLanguagePairDataset(LanguagePairDataset):
    """
    A pair of torch.utils.data.Datasets.
    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
        langpair_dataset (LanguagePairDataset, optional)
        feat (numpy.ndarray)
    """

    def __init__(
        self,
        langpair_dataset: LanguagePairDataset,
        feat_num: int,
        feat_channel: int,
        feat: torch.Tensor,
        feat_padding_mask: torch.Tensor = None,
        feat_indices: List[int] = None,
        enable_tgt_lang_tok: bool = False,
        max_items: int = None,
    ):
        self.__dict__.update(langpair_dataset.__dict__)

        self.feat_num = feat_num
        self.feat_channel = feat_channel
        self.enable_tgt_lang_tok = enable_tgt_lang_tok

        self.feat = self.build_feat_data(feat)

        # feature padding mask (bsz, feat_num)
        if feat_padding_mask is None:
            # no padding
            self.feat_padding = torch.zeros(len(self.src), self.feat_num).bool()
        else:
            self.feat_padding = feat_padding_mask.unsqueeze(-1).repeat(1, feat_num)

        # feature indices
        # set this argument to avoid duplicating a same feature paired to multiple sentences
        self.feat_indices = feat_indices

        self.max_items = int(max_items) if max_items is not None else None

    def build_feat_data(self, feat):

        ## sanity check
        assert feat.dim() in (2, 3, 4), "this feature file is incompatible."

        ## convert feature into (time, bsz, channels)
        if feat.dim() == 2:             # global feat: (bsz, channels)
            feat = feat.unsqueeze(0)
        elif feat.dim() == 3: # faster-rcnn: (bsz, channels, feat_num)
            feat = feat.permute(2, 0, 1)
        elif feat.dim() == 4: # resnet local feat: (bsz, channels, feat_num_1, feat_num_2)
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1).permute(2, 0, 1)

        assert feat.shape[0] >= self.feat_num
        assert feat.shape[2] == self.feat_channel
        
        # shrink
        feat = feat[:self.feat_num, :, :]

        return feat

    def get_feat_index(self, index):
        if self.feat_indices is None:
            return index
        else:
            return self.feat_indices[index]

    def __len__(self):
        if self.max_items is None:
            return len(self.src)
        else:
            return min(len(self.src), self.max_items)

    def __getitem__(self, index):
        example = super().__getitem__(index)

        feat_index = self.get_feat_index(index)
        feat = self.feat.select(1, feat_index)
        feat_padding_mask = self.feat_padding.select(0, feat_index)

        example['gold_feat'] = feat
        example['feat'] = feat.masked_fill(feat_padding_mask.unsqueeze(-1), 0.)
        example['feat_padding_mask'] = feat_padding_mask

        if self.enable_tgt_lang_tok:
            example['tgt_lang_tok'] = self.tgt.token

        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.
        Returns:
            dict: a mini-batch with the following keys:
                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:
                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        # get sort_order before running super().collater
        pad_idx = self.src_dict.pad()
        _, sort_order = torch.LongTensor(
            [s["source"].ne(pad_idx).long().sum() for s in samples]
        ).sort(descending=True)

        # langpair
        res = super().collater(samples, pad_to_length)

        # gold_feat
        gold_feat_data = torch.stack([item['gold_feat'] for item in samples], dim=1)
        gold_feat_data = gold_feat_data.index_select(1, sort_order)
        res['gold_feat'] = gold_feat_data

        # feat
        feat_data = torch.stack([item['feat'] for item in samples], dim=1)
        feat_data = feat_data.index_select(1, sort_order)
        res['net_input']['feat'] = feat_data

        feat_padding_data = torch.stack([item['feat_padding_mask'] for item in samples], dim=0)
        feat_padding_data = feat_padding_data.index_select(0, sort_order)
        res['net_input']['feat_padding_mask'] = feat_padding_data

        if self.enable_tgt_lang_tok:
            tgt_lang_tok_data = torch.LongTensor([item['tgt_lang_tok'] for item in samples])
            tgt_lang_tok_data = tgt_lang_tok_data.index_select(0, sort_order)
            res['net_input']['tgt_lang_tok'] = tgt_lang_tok_data

        return res