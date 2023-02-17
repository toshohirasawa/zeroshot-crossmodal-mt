import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils
import tensorpack.dataflow as td

logger = logging.getLogger(__name__)

class PreprocessDataBatch(object):
    def __init__(
        self,
        tokenizer,
    ) -> None:
        self.tokenizer = tokenizer

    def __call__(self, data):
        image_feature_wp, image_cls_wp, obj_labels, obj_confs, attr_labels, attr_confs, attr_scores, \
            image_location_wp, num_boxes, image_h, image_w, image_id, caption = data
        
        image_feature = np.zeros((max(num_boxes), 2048), dtype=np.float32)
        # image_cls = np.zeros((max(num_boxes), 1601), dtype=np.float32)
        # image_attrs = np.zeros((max(num_boxes), 401), dtype=np.float32)
        # image_location = np.zeros((max(num_boxes), self.num_locs), dtype=np.float32)
        tokens_caption = self.tokenizer.encode(caption)

        # calculate the IOU here.
        # overlaps = iou(image_location_wp, image_location_wp)

        num_boxes = int(num_boxes)
        image_feature[:num_boxes] = image_feature_wp
        # image_cls[:num_boxes] = image_cls_wp
        # image_attrs[:num_boxes] = attr_scores
        # image_location[:num_boxes, :4] = image_location_wp
        # obj_labels = obj_labels[:num_boxes]
        # obj_confs = obj_confs[:num_boxes]
        # attr_labels = attr_labels[:num_boxes]
        # attr_confs = attr_confs[:num_boxes]

        # if self.num_locs >= 5:
        #     image_location[:, -1] = (
        #         (image_location[:, 3] - image_location[:, 1])
        #         * (image_location[:, 2] - image_location[:, 0])
        #         / (float(image_w) * float(image_h))
        #     )

        # # Normalize the box locations (to 0 ~ 1)
        # image_location[:, 0] = image_location[:, 0] / float(image_w)
        # image_location[:, 1] = image_location[:, 1] / float(image_h)
        # image_location[:, 2] = image_location[:, 2] / float(image_w)
        # image_location[:, 3] = image_location[:, 3] / float(image_h)

        # if self.num_locs > 5:
        #     image_location[:, 4] = image_location[:, 2] - image_location[:, 0]
        #     image_location[:, 5] = image_location[:, 3] - image_location[:, 1]

        # caption, label = self.random_cap(caption)
        # tokens_caption = self.tokenizer.encode(caption, add_special_tokens=False)

        # cur_example = InputExample(
        #     image_feat=image_feature,
        #     image_cls=image_cls,
        #     obj_labels=obj_labels,
        #     obj_confs=obj_confs,
        #     attr_labels=attr_labels,
        #     attr_confs=attr_confs,
        #     image_attrs=image_attrs,
        #     caption=tokens_caption,
        #     is_next=label,
        #     image_loc=image_location,
        #     num_boxes=num_boxes,
        #     overlaps=overlaps,
        # )

        # # transform sample to features
        # cur_features = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.region_len)

        # cur_tensors = (
        #     cur_features.input_ids,
        #     cur_features.input_mask,
        #     cur_features.segment_ids,
        #     cur_features.lm_label_ids,
        #     cur_features.is_next,
        #     cur_features.image_feat,
        #     cur_features.image_loc,
        #     cur_features.image_cls,
        #     cur_features.obj_labels,
        #     cur_features.obj_confs,
        #     cur_features.attr_labels,
        #     cur_features.attr_confs,
        #     cur_features.image_attrs,
        #     cur_features.image_label,
        #     cur_features.image_mask,
        #     cur_features.masked_label,
        #     image_id,
        # )
        return (
            # num_boxes,
            image_feature,
            # image_cls,
            # image_attrs,
            # image_location,
            # obj_labels,
            # obj_confs,
            # attr_labels,
            # attr_confs,
            tokens_caption,
        )

class ConceptCapVoltaDataset(FairseqDataset):
    def __init__(
        self,
        annotation_file: str,
        lmdb_file: str,
        tokenizer,

        cache=10000,
        num_workers=5,
        batch_size=512,
    ) -> None:
        super().__init__()

        preprocess_function = PreprocessDataBatch(
            tokenizer=tokenizer
        )

        ds = td.LMDBSerializer.load(lmdb_file)
        ds = td.LocallyShuffleData(ds, cache)
        ds = td.PrefetchData(ds, 5000, 1)
        ds = td.MapData(ds, preprocess_function)
        ds = td.PrefetchDataZMQ(ds, num_workers)
        ds = td.BatchData(ds, batch_size)
        ds.reset_state()

        self.ds = ds
        self.stream = self.ds.get_data()
        self.batch_index = -1
        self.queue = []

    def __getitem__(self, index):
        if len(self.queue) == 0:

            if self.batch_index == len(self.ds) - 1:
                # run out batches, reload
                self.ds.reset_state()
                self.stream = self.ds.get_data()
                self.batch_index = -1
            
            self.queue = next(self.stream)
            self.batch_index += 1

        
        image_feature, caption = [q.pop() for q in self.queue]

        return {
            "feature": image_feature,
            "caption": caption,
        }

    def __len__(self):
        return len(self.ds)


if __name__ == '__main__':
    import os, sys

    ds = ConceptCapVoltaDataset(
        annotation_file=sys.argv[1],
        lmdb_file=sys.argv[2]
    )

    print(f'loaded {len(ds):,} records.')

    print(ds[0])
    print(ds[1])
    print(ds[2])