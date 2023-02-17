# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from .label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion
)

class MaxMarginLoss(torch.nn.Module):
    """
    Ranking-based max-margin loss function with negative sampling
    for sequential continuous representations.
    - draw negative samples from the samples in the same batch and at the same position
    """

    def __init__(self, margin: float):
        super().__init__()

        assert margin > 0., "margin must be > 0."

        self.margin = margin

    def forward(self, 
        pred: torch.Tensor, 
        gold: torch.Tensor, 
        padding_mask: Optional[torch.Tensor] = None
    ):
        """
        pred (time x bsz x channel): system output
        gold (time x bsz x channel): ground truth
        padding_mask (bsz x time): Tensor that indicates whether a element is a padded one
        """
        if padding_mask is None:
            feat_num, bsz = pred.shape[:2]
            padding_mask = torch.zeros(pred.bsz, feat_num).type_as(torch.bool).to(gold.device)

        # at least one position should have 2+ non-pad element
        # as max-margin loss requires at least two elements to compute
        if (padding_mask is not None) and (((~padding_mask).sum(dim=0) > 1).sum() == 0):
            return torch.zeros([]).type_as(gold).to(gold.device)

        bsz = pred.shape[1]

        # for simplifying cosine distance calculation below
        normed_pred = pred / pred.norm(dim=-1, keepdim=True)

        # extract and normalize feats
        normed_gold = gold / gold.norm(dim=-1, keepdim=True)

        # Implementation from original paper (Max-Margin)
        errors = torch.bmm(normed_pred, normed_gold.transpose(-2,-1))
        preds = errors.diagonal(dim1=-2, dim2=-1)

        # all contrastive images for each sentence
        loss_s = self.margin - errors + preds.unsqueeze(-1)
        loss_s = torch.max(loss_s, torch.zeros_like(loss_s))

        # all contrastive sentences for each image
        loss_i = self.margin - errors + preds.unsqueeze(-2)
        loss_i = torch.max(loss_i, torch.zeros_like(loss_i))

        # total loss
        losses = loss_s + loss_i
        losses[:, range(bsz), range(bsz)] = 0.0

        # mask-out loss on pad elements
        if padding_mask is not None:
            losses[padding_mask.transpose(0, 1)] = 0.0
            losses.transpose_(1, 2)[padding_mask.transpose(0, 1)] = 0.0
            # need not re-transpose as axis is arbitary for following computing

        # mean over non-zero loss if having losses
        final_loss = losses[losses != 0.0].mean()

        # one for positive sample
        return final_loss

@dataclass
class LabelSmoothedCrossEntropyCriterionWithMaxMarginConfig(LabelSmoothedCrossEntropyCriterionConfig):
    max_margin_weight: float = field(
        default=1.0, 
        metadata={"help": "weight for CTC loss"}
    )
    margin: float = field(
        default=0.1, 
        metadata={"help": "margin for max-margin loss"}
    )

@register_criterion(
    "label_smoothed_cross_entropy_with_max_margin", 
    dataclass=LabelSmoothedCrossEntropyCriterionWithMaxMarginConfig
)
class LabelSmoothedCrossEntropyWithMaxMarginCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size = 0,
        report_accuracy = False,
        margin: float = 0.1,
        max_margin_weight: float = 1.0,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)

        self.margin = margin
        self.max_margin_weight = max_margin_weight

        self.max_margin_loss = MaxMarginLoss(self.margin)


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss, max_margin_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "max_margin_loss": max_margin_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        loss, nll_loss = super().compute_loss(model, net_output, sample, reduce)

        _, extra = net_output
        max_margin_loss = torch.tensor(0.0).type_as(loss)
        if self.max_margin_weight > 0:
            feat_gold = sample["gold_feat"]
            feat_padding_mask = sample["net_input"]['feat_padding_mask']
            feat_pred = extra["feat_pred"]

            max_margin_loss = self.max_margin_loss(
                gold = feat_gold,
                pred = feat_pred,
                padding_mask = feat_padding_mask,
            ) * self.max_margin_weight

        loss += max_margin_loss

        return loss, nll_loss, max_margin_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)

        loss_sum = sum(log.get("max_margin_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("max_margin_loss", loss_sum, sample_size, round=3)
