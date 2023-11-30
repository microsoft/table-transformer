# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy
import time
import os
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.utils.rnn import pad_sequence
import pprint

from util.box_ops import box_cxcywh_to_xyxy, box_cxcywh_to_xyxy_with_bounds, generalized_box_iou, cross_bounded_box_iou_with_bounds
from util import box_ops

def robust_linear_sum_assignment(c):
    try:
        return linear_sum_assignment(c)
    except ValueError as e:
        d = numpy.nan_to_num(c, nan=1e20, posinf=1e10, neginf=-1e10)
        print("Exception {} with input {}. Trying again with input {}.".format(e, c, d))
        return linear_sum_assignment(d)


def last_step(cs):
    # https://github.com/scipy/scipy/blob/main/scipy/optimize/rectangular_lsap/rectangular_lsap.cpp
    return [
        robust_linear_sum_assignment(c)
        for c in cs
    ]


def stable_last_step(cs):
    # https://github.com/scipy/scipy/blob/main/scipy/optimize/rectangular_lsap/rectangular_lsap.cpp
    # Note: argsort_iter uses a non-stable sort.
    result = [
        tuple(
            zip(
                *[
                    (e[1][1], e[0])
                    for e in sorted(
                        enumerate(
                            list(
                                zip(*robust_linear_sum_assignment(c))
                            )
                        ),
                        key=lambda e: e[1][1],
                    )
                ]
            )
        )
        or ((), ())
        for c in cs
    ]
    return result


def _core_old_hungarian_matching(
    outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor
):
    bs, num_queries = outputs["pred_logits"].shape[:2]
    # We flatten to compute the cost matrices in a batch
    out_prob = (
        outputs["pred_logits"].flatten(0, 1).softmax(-1)
    )  # [batch_size * num_queries, num_classes]
    out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

    # Also concat the target labels and boxes
    tgt_ids = torch.cat([v["labels"] for v in targets])
    tgt_bbox = torch.cat([v["boxes"] for v in targets])

    # Compute the classification cost. Contrary to the loss, we don't use the Negative Log Likelihood,
    # but approximate it in 1 - proba[target class].
    # The 1 is a constant that doesn't change the matching, it can be ommitted.
    cost_class = -out_prob[:, tgt_ids]  # [batch_size * num_queries, sum(sizes)]

    # Compute the L1 cost between boxes
    cost_bbox = torch.cdist(
        out_bbox, tgt_bbox, p=1
    )  # [batch_size * num_queries, sum(sizes)]

    # Compute the giou cost betwen boxes
    cost_giou = -generalized_box_iou(
        box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
    )  # [batch_size * num_queries, sum(sizes)]

    # Final cost matrix
    C = (
        cost_bbox_factor * cost_bbox
        + cost_class_factor * cost_class
        + cost_giou_factor * cost_giou
    )
    C = C.view(bs, num_queries, -1).cpu()

    sizes = [len(v["boxes"]) for v in targets]
    return [c[i] for i, c in enumerate(C.split(sizes, -1))]


def _old_hungarian_matching(
    outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor
):
    return last_step(
        _core_old_hungarian_matching(outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor))


def _core_new_hungarian_matching(
    outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor
):
    optimize_batch_size_one = True
    if len(targets) == 1 and optimize_batch_size_one:
        tgt_ids = targets[0]["labels"].unsqueeze(dim=0)
        tgt_bbox = targets[0]["boxes"].unsqueeze(dim=0)
    else:
        # https://github.com/pytorch/pytorch/blob/5044d9dc517dd3ad547334f26d102a9d454b8f59/torch/csrc/api/include/torch/nn/utils/rnn.h#L279
        tgt_ids = pad_sequence(
            [target["labels"] for target in targets], batch_first=True, padding_value=0
        )
        tgt_bbox = pad_sequence(
            [target["boxes"] for target in targets], batch_first=True, padding_value=0
        )
    out_prob = outputs["pred_logits"].softmax(
        -1
    )  # [batch_size, num_queries, num_classes]
    cost_class = -out_prob.gather(
        dim=2, index=tgt_ids.unsqueeze(dim=1).expand(-1, out_prob.shape[1], -1)
    )  # [batch_size, num_queries, max_size]
    out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 4]
    cost_bbox = torch.cdist(
        out_bbox, tgt_bbox, p=1
    )  # [batch_size, num_queries, max_size]
    out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
    tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
    cost_giou = -generalized_box_iou(
        out_bbox_xyxy, tgt_bbox_xyxy
    )  # [batch_size, num_queries, max_size]
    C = (
        cost_bbox_factor * cost_bbox
        + cost_class_factor * cost_class
        + cost_giou_factor * cost_giou
    )  # [batch_size, num_queries, max_size]
    c = C.cpu()
    return [c[i, :, : len(v["labels"])] for i, v in enumerate(targets)]

def _new_hungarian_matching(
    outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor
):
    return last_step(
        _core_new_hungarian_matching(outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor))


def _core_mem_hungarian_matching(
    outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor
):
    optimize_batch_size_one = True
    if len(targets) == 1 and optimize_batch_size_one:
        tgt_ids = targets[0]["labels"].unsqueeze(dim=0)
        tgt_bbox = targets[0]["boxes"].unsqueeze(dim=0)
    else:
        # https://github.com/pytorch/pytorch/blob/5044d9dc517dd3ad547334f26d102a9d454b8f59/torch/csrc/api/include/torch/nn/utils/rnn.h#L279
        tgt_ids = pad_sequence(
            [target["labels"] for target in targets], batch_first=True, padding_value=0
        )
        tgt_bbox = pad_sequence(
            [target["boxes"] for target in targets], batch_first=True, padding_value=0
        )
    out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 4]
    cost_bbox = torch.cdist(
        out_bbox, tgt_bbox, p=1
    )  # [batch_size, num_queries, max_size]
    out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
    tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
    cost_giou = -generalized_box_iou(
        out_bbox_xyxy, tgt_bbox_xyxy
    )  # [batch_size, num_queries, max_size]
    C = (
        cost_bbox_factor * cost_bbox + cost_giou_factor * cost_giou
    )  # [batch_size, num_queries, max_size]
    out_prob = outputs["pred_logits"].softmax(
        -1
    )  # [batch_size, num_queries, num_classes + 1]
    # Reuse cost_giou's storage.
    cost_class = torch.gather(
        out_prob,
        dim=2,
        index=tgt_ids.unsqueeze(dim=1).expand(-1, out_prob.shape[1], -1),
        out=cost_giou,
    )  # [batch_size, num_queries, max_size]
    C -= cost_class_factor * cost_class
    c = C.cpu()
    return [c[i, :, : len(v["labels"])] for i, v in enumerate(targets)]

def _mem_hungarian_matching(
    outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor
):
    return last_step(
        _core_mem_hungarian_matching(outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor))


def _core_final_hungarian_matching(
    outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor, enable_bounds
):
    # In the comments below:
    # - `bs` is the batch size, i.e. outputs["pred_logits"].shape[0];
    # - `mo` is the maximum number of objects over all the targets,
    # i.e. `max((len(v["labels"]) for v in targets))`;
    # - `q` is the number of queries, i.e. outputs["pred_logits"].shape[1];
    # - `cl` is the number of classes including no-object,
    # i.e. outputs["pred_logits"].shape[2] or self.num_classes + 1.
    if len(targets) == 1:
        # This branch is just an optimization, not needed for correctness.
        tgt_ids = targets[0]["labels"].unsqueeze(dim=0)
        tgt_bbox = targets[0]["boxes"].unsqueeze(dim=0)
    else:
        tgt_ids = pad_sequence(
            [target["labels"] for target in targets],
            batch_first=True,
            padding_value=0
        )  # (bs, mo)
        tgt_bbox = pad_sequence(
            [target["boxes"] for target in targets],
            batch_first=True,
            padding_value=0
        )  # (bs, mo, 4)
    # print("tgt_bbox: {}".format(tgt_bbox))
    out_bbox = outputs["pred_boxes"]  # (bs, q, 4)
    # print("out_bbox: {}".format(out_bbox))
    cost_bbox = (box_ops.cross_lp_loss_with_bounds_cxcywh if enable_bounds else torch.cdist)(out_bbox, tgt_bbox, p=1)  # (bs, q, mo)
    # print("cost_bbox: {}".format(cost_bbox))
    out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
    tgt_bbox_xyxy = box_cxcywh_to_xyxy_with_bounds(tgt_bbox) if enable_bounds else box_cxcywh_to_xyxy(tgt_bbox)
    # print("out_bbox_xyxy: {}".format(out_bbox_xyxy))
    # print("tgt_bbox_xyxy: {}".format(tgt_bbox_xyxy))
    giou = (cross_bounded_box_iou_with_bounds if enable_bounds else generalized_box_iou)(
        out_bbox_xyxy, tgt_bbox_xyxy)  # (bs, q, mo)
    # print("giou: {}".format(giou))
    out_prob = outputs["pred_logits"].softmax(-1)  # (bs, q, c)
    # Compute the classification cost. Contrary to the loss, we don't use
    # the Negative Log Likelihood, but approximate it
    # in `1 - proba[target class]`. The 1 is a constant that does not
    # change the matching, it can be ommitted.
    prob_class = torch.gather(
        out_prob,
        dim=2,
        index=tgt_ids.unsqueeze(dim=1).expand(-1, out_prob.shape[1], -1)
    )  # (bs, q, mo)
    # print("prob_class: {}".format(prob_class))
    C = cost_bbox_factor * cost_bbox - cost_giou_factor * giou - cost_class_factor * prob_class
    c = C.cpu()
    # print("c: {}".format(c))
    return [c[i, :, : len(v["labels"])] for i, v in enumerate(targets)]

def _final_hungarian_matching(
    outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor, enable_bounds
):
    return last_step(
        _core_final_hungarian_matching(outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor, enable_bounds))

def _core_mem_hungarian_matching(
    outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor
):
    if len(targets) == 1:
        tgt_ids = targets[0]["labels"].unsqueeze(dim=0)
        tgt_bbox = targets[0]["boxes"].unsqueeze(dim=0)
    else:
        # https://github.com/pytorch/pytorch/blob/5044d9dc517dd3ad547334f26d102a9d454b8f59/torch/csrc/api/include/torch/nn/utils/rnn.h#L279
        tgt_ids = pad_sequence(
            [target["labels"] for target in targets], batch_first=True, padding_value=0
        )
        tgt_bbox = pad_sequence(
            [target["boxes"] for target in targets], batch_first=True, padding_value=0
        )
    out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 4]
    cost_bbox = torch.cdist(
        out_bbox, tgt_bbox, p=1
    )  # [batch_size, num_queries, max_size]
    out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
    tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
    giou = generalized_box_iou(
        out_bbox_xyxy, tgt_bbox_xyxy
    )  # [batch_size, num_queries, max_size]
    C = (
        cost_bbox_factor * cost_bbox - cost_giou_factor * giou
    )  # [batch_size, num_queries, max_size]
    out_prob = outputs["pred_logits"].softmax(
        -1
    )  # [batch_size, num_queries, num_classes + 1]
    # Reuse giou's storage.
    prob_class = torch.gather(
        out_prob,
        dim=2,
        index=tgt_ids.unsqueeze(dim=1).expand(-1, out_prob.shape[1], -1),
        out=giou,
    )  # [batch_size, num_queries, max_size]
    C -= cost_class_factor * prob_class
    c = C.cpu()
    return [c[i, :, : len(v["labels"])] for i, v in enumerate(targets)]

def _inplace_hungarian_matching(
    outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor
):
    return last_step(
        _core_mem_hungarian_matching(outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor))

def _core_inplace_hungarian_matching(
    outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor
):
    if len(targets) == 1:
        tgt_ids = targets[0]["labels"].unsqueeze(dim=0)
        tgt_bbox = targets[0]["boxes"].unsqueeze(dim=0)
    else:
        # https://github.com/pytorch/pytorch/blob/5044d9dc517dd3ad547334f26d102a9d454b8f59/torch/csrc/api/include/torch/nn/utils/rnn.h#L279
        tgt_ids = pad_sequence(
            [target["labels"] for target in targets], batch_first=True, padding_value=0
        )
        tgt_bbox = pad_sequence(
            [target["boxes"] for target in targets], batch_first=True, padding_value=0
        )
    out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 4]
    C = torch.cdist(
        out_bbox, tgt_bbox, p=1
    )  # [batch_size, num_queries, max_size]
    C *= cost_bbox_factor
    out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
    tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
    giou = generalized_box_iou(
        out_bbox_xyxy, tgt_bbox_xyxy
    )  # [batch_size, num_queries, max_size]
    giou *= cost_giou_factor
    C -= giou
    out_prob = outputs["pred_logits"].softmax(
        -1
    )  # [batch_size, num_queries, num_classes + 1]
    # Reuse giou's storage.
    prob_class = torch.gather(
        out_prob,
        dim=2,
        index=tgt_ids.unsqueeze(dim=1).expand(-1, out_prob.shape[1], -1),
        out=giou,
    )  # [batch_size, num_queries, max_size]
    prob_class *= cost_class_factor
    C -= prob_class
    c = C.cpu()
    return [c[i, :, : len(v["labels"])] for i, v in enumerate(targets)]


def _core_tr_hungarian_matching(
    outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor
):
    optimize_batch_size_one = True
    if len(targets) == 1 and optimize_batch_size_one:
        tgt_ids = targets[0]["labels"].unsqueeze(dim=0)
        tgt_bbox = targets[0]["boxes"].unsqueeze(dim=0)
    else:
        # https://github.com/pytorch/pytorch/blob/5044d9dc517dd3ad547334f26d102a9d454b8f59/torch/csrc/api/include/torch/nn/utils/rnn.h#L279
        tgt_ids = pad_sequence(
            [target["labels"] for target in targets], batch_first=True, padding_value=0
        )
        tgt_bbox = pad_sequence(
            [target["boxes"] for target in targets], batch_first=True, padding_value=0
        )
    out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 8 if enable_bounds else 4]
    cost_bbox = torch.cdist(
        out_bbox, tgt_bbox, p=1
    )  # [batch_size, num_queries, max_size]
    out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
    tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
    giou = generalized_box_iou(
        out_bbox_xyxy, tgt_bbox_xyxy
    )  # [batch_size, num_queries, max_size]
    C = (
        cost_bbox_factor * cost_bbox - cost_giou_factor * giou
    )  # [batch_size, num_queries, max_size]
    out_prob = outputs["pred_logits"].softmax(
        -1
    )  # [batch_size, num_queries, num_classes]
    # Reuse giou's storage.
    prob_class = torch.gather(
        out_prob,
        dim=2,
        index=tgt_ids.unsqueeze(dim=1).expand(-1, out_prob.shape[1], -1),
        out=giou,
    )  # [batch_size, num_queries, max_size]
    C -= cost_class_factor * prob_class
    c = C.transpose(1, 2).cpu()
    return [c[i, : len(v["labels"]), :] for i, v in enumerate(targets)]

def _tr_hungarian_matching(
    outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor
):
    return stable_last_step(
        _core_tr_hungarian_matching(outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor))


def _test_hungarian_matching(
    outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor
):
    a = [
        f(outputs, targets, cost_bbox_factor, cost_class_factor, cost_giou_factor)
        for f in (
            _core_old_hungarian_matching,
            _core_new_hungarian_matching,
            _core_mem_hungarian_matching,
            _core_final_hungarian_matching,
            _core_inplace_hungarian_matching,
            _core_tr_hungarian_matching,
        )
    ]
    x = a[0]
    for y in a[1:-1]:
        assert len(x) == len(y), (x, y)
        for mx, my in zip(x, y):
            assert mx.shape == my.shape, (mx.shape, my.shape)
            assert numpy.allclose(mx, my)
    z = a[-1]
    assert len(x) == len(z), (x, z)
    for mx, mz in zip(x, z):
        assert mx.shape == torch.Size(reversed(mz.shape)), (mx.shape, mz.shape)
        assert numpy.allclose(mx, mz.transpose(0, 1))

            # assert numpy.array_equal(mx, my), (mx, my)
    return last_step(x)

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, cost_class: float, cost_bbox: float, cost_giou: float, enable_bounds: bool
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.enable_bounds = enable_bounds
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"
        match os.environ.get("HUNGARIAN_MATCH_ALGO"):
            case "tr":
                self.algo = _tr_hungarian_matching
            case "mem":
                self.algo = _mem_hungarian_matching
            case "final":
                self.algo = _final_hungarian_matching
            case "inplace":
                self.algo = _inplace_hungarian_matching
            case "new":
                self.algo = _new_hungarian_matching
            case "old":
                self.algo = _old_hungarian_matching
            case "test":
                self.algo = _test_hungarian_matching
            case _:
                self.algo = _final_hungarian_matching

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # print("outputs: {} targets: {}", pprint.pformat(outputs), pprint.pformat(targets))
        indices = self.algo(
            outputs, targets, self.cost_bbox, self.cost_class, self.cost_giou, self.enable_bounds
        )
        # print("indices: {}".format(indices))
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        enable_bounds=args.enable_bounds
    )
