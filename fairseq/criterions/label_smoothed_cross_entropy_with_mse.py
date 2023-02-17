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

@dataclass
class LabelSmoothedCrossEntropyCriterionWithMSEConfig(LabelSmoothedCrossEntropyCriterionConfig):
    pass

@register_criterion(
    "label_smoothed_cross_entropy_with_mse", 
    dataclass=LabelSmoothedCrossEntropyCriterionWithMSEConfig
)
class LabelSmoothedCrossEntropyWithMSECriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size = 0,
        report_accuracy = False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)

        self.mse_loss = torch.nn.MSELoss()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss, mse_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "mse_loss": mse_loss.data,
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
        
        feat = sample["net_input"]['feat']
        feat_padding_mask = sample["net_input"]['feat_padding_mask']

        reconst_feat = extra["reconst_feat"][0]

        mse_loss = self.mse_loss(feat, reconst_feat)
        loss += mse_loss

        return loss, nll_loss, mse_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)

        loss_sum = sum(log.get("mse_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("mse_loss", loss_sum, sample_size, round=3)
