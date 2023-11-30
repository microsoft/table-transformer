# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import unittest

import torch
import functools
import operator
import pprint

from models.matcher import HungarianMatcher
from models.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from models.backbone import Backbone
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
from hubconf import detr_resnet50, detr_resnet50_panoptic
from itertools import combinations_with_replacement
from torchvision import ops
from torch.testing import assert_close
import numpy as np
from torch import nn
from torch import linalg as LA
from torch.distributions.uniform import Uniform

# onnxruntime requires python 3.5 or above
try:
    import onnxruntime
except ImportError:
    onnxruntime = None


def _cross_dist_with_bounds(pred, target, *, p):
    b = pred.to(torch.float32)
    x = box_ops.cross_dist_with_bounds_neighbour(b, target)
    return LA.vector_norm(x - b[..., None, :], ord=p, dim=-1)


def make_present(c):
    return torch.cat([torch.min(c[:2], c[2:]), torch.max(c[:2], c[2:])])


class Tester(unittest.TestCase):
    def test_box_cxcywh_to_xyxy(self):
        t = torch.rand(10, 4)
        r = box_ops.box_xyxy_to_cxcywh(box_ops.box_cxcywh_to_xyxy(t))
        self.assertLess((t - r).abs().max(), 1e-5)

    @staticmethod
    def indices_torch2python(indices):
        return [(i.tolist(), j.tolist()) for i, j in indices]

    def test_hungarian(self):
        n_queries, n_targets, n_classes = 100, 15, 91
        logits = torch.rand(1, n_queries, n_classes + 1)
        boxes = torch.rand(1, n_queries, 4)
        tgt_labels = torch.randint(high=n_classes, size=(n_targets,))
        tgt_boxes = torch.rand(n_targets, 4)
        matcher = HungarianMatcher(1, 1, 1, False)
        targets = [{"labels": tgt_labels, "boxes": tgt_boxes}]
        indices_single = matcher({"pred_logits": logits, "pred_boxes": boxes}, targets)
        batch_size = 4
        indices_batched = matcher(
            {
                "pred_logits": logits.repeat(batch_size, 1, 1),
                "pred_boxes": boxes.repeat(batch_size, 1, 1),
            },
            targets * batch_size,
        )
        self.assertEqual(len(indices_single[0][0]), n_targets)
        self.assertEqual(len(indices_single[0][1]), n_targets)
        for i in range(batch_size):
            self.assertEqual(
                self.indices_torch2python(indices_single),
                self.indices_torch2python([indices_batched[i]]),
            )

        # test with empty targets
        tgt_labels_empty = torch.randint(high=n_classes, size=(0,))
        tgt_boxes_empty = torch.rand(0, 4)
        targets_empty = [{"labels": tgt_labels_empty, "boxes": tgt_boxes_empty}]
        indices = matcher(
            {
                "pred_logits": logits.repeat(2, 1, 1),
                "pred_boxes": boxes.repeat(2, 1, 1),
            },
            targets + targets_empty,
        )
        self.assertEqual(len(indices[1][0]), 0)
        indices = matcher(
            {
                "pred_logits": logits.repeat(2, 1, 1),
                "pred_boxes": boxes.repeat(2, 1, 1),
            },
            targets_empty * 2,
        )
        self.assertEqual(len(indices[0][0]), 0)

    def test_position_encoding_script(self):
        m1, m2 = PositionEmbeddingSine(), PositionEmbeddingLearned()
        mm1, mm2 = torch.jit.script(m1), torch.jit.script(m2)  # noqa

    def test_backbone_script(self):
        backbone = Backbone("resnet50", True, False, False)
        torch.jit.script(backbone)  # noqa

    def test_model_script_detection(self):
        model = detr_resnet50(pretrained=False).eval()
        scripted_model = torch.jit.script(model)
        x = nested_tensor_from_tensor_list(
            [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
        )
        out = model(x)
        out_script = scripted_model(x)
        self.assertTrue(out["pred_logits"].equal(out_script["pred_logits"]))
        self.assertTrue(out["pred_boxes"].equal(out_script["pred_boxes"]))

    def test_model_script_panoptic(self):
        model = detr_resnet50_panoptic(pretrained=False).eval()
        scripted_model = torch.jit.script(model)
        x = nested_tensor_from_tensor_list(
            [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
        )
        out = model(x)
        out_script = scripted_model(x)
        self.assertTrue(out["pred_logits"].equal(out_script["pred_logits"]))
        self.assertTrue(out["pred_boxes"].equal(out_script["pred_boxes"]))
        self.assertTrue(out["pred_masks"].equal(out_script["pred_masks"]))

    def test_model_detection_different_inputs(self):
        model = detr_resnet50(pretrained=False).eval()
        # support NestedTensor
        x = nested_tensor_from_tensor_list(
            [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
        )
        out = model(x)
        self.assertIn("pred_logits", out)
        # and 4d Tensor
        x = torch.rand(1, 3, 200, 200)
        out = model(x)
        self.assertIn("pred_logits", out)
        # and List[Tensor[C, H, W]]
        x = torch.rand(3, 200, 200)
        out = model([x])
        self.assertIn("pred_logits", out)

    def test_box_iou_multiple_dimensions(self):
        for extra_dims in range(3):
            for extra_lengths in combinations_with_replacement(range(1, 4), extra_dims):
                p = functools.reduce(operator.mul, extra_lengths, 1)
                for n in range(3):
                    a = torch.rand(extra_lengths + (n, 4))
                    for m in range(3):
                        b = torch.rand(extra_lengths + (m, 4))
                        iou, union = box_ops.cross_box_iou(a, b)
                        self.assertTupleEqual(iou.shape, union.shape)
                        self.assertTupleEqual(iou.shape, extra_lengths + (n, m))
                        iou_it = iter(iou.view(p, n, m))
                        for x, y in zip(a.view(p, n, 4), b.view(p, m, 4), strict=True):
                            self.assertTrue(
                                torch.equal(next(iou_it), ops.box_iou(x, y))
                            )

    def test_generalized_box_iou_multiple_dimensions(self):
        a = torch.tensor([1, 1, 2, 2])
        b = torch.tensor([1, 2, 3, 5])
        ab = -0.1250
        self.assertTrue(
            torch.equal(
                box_ops.generalized_box_iou(a[None, :], b[None, :]),
                torch.Tensor([[ab]]),
            )
        )
        self.assertTrue(
            torch.equal(
                box_ops.generalized_box_iou(a[None, None, :], b[None, None, :]),
                torch.Tensor([[[ab]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                box_ops.generalized_box_iou(
                    a[None, None, None, :], b[None, None, None, :]
                ),
                torch.Tensor([[[[ab]]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                box_ops.generalized_box_iou(
                    torch.stack([a, a, b, b]), torch.stack([a, b])
                ),
                torch.Tensor(torch.Tensor([[1, ab], [1, ab], [ab, 1], [ab, 1]])),
            )
        )


@unittest.skipIf(onnxruntime is None, "ONNX Runtime unavailable")
class ONNXExporterTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)

    def run_model(
        self,
        model,
        inputs_list,
        tolerate_small_mismatch=False,
        do_constant_folding=True,
        dynamic_axes=None,
        output_names=None,
        input_names=None,
    ):
        model.eval()

        onnx_io = io.BytesIO()
        # export to onnx with the first input
        torch.onnx.export(
            model,
            inputs_list[0],
            onnx_io,
            do_constant_folding=do_constant_folding,
            opset_version=12,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            output_names=output_names,
        )
        # validate the exported model with onnx runtime
        for test_inputs in inputs_list:
            with torch.no_grad():
                if isinstance(test_inputs, torch.Tensor) or isinstance(
                    test_inputs, list
                ):
                    test_inputs = (nested_tensor_from_tensor_list(test_inputs),)
                test_ouputs = model(*test_inputs)
                if isinstance(test_ouputs, torch.Tensor):
                    test_ouputs = (test_ouputs,)
            self.ort_validate(
                onnx_io, test_inputs, test_ouputs, tolerate_small_mismatch
            )

    def ort_validate(self, onnx_io, inputs, outputs, tolerate_small_mismatch=False):
        inputs, _ = torch.jit._flatten(inputs)
        outputs, _ = torch.jit._flatten(outputs)

        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.cpu().numpy()

        inputs = list(map(to_numpy, inputs))
        outputs = list(map(to_numpy, outputs))

        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
        # compute onnxruntime output prediction
        ort_inputs = dict(
            (ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs)
        )
        ort_outs = ort_session.run(None, ort_inputs)
        for i in range(0, len(outputs)):
            try:
                torch.testing.assert_allclose(
                    outputs[i], ort_outs[i], rtol=1e-03, atol=1e-05
                )
            except AssertionError as error:
                if tolerate_small_mismatch:
                    self.assertIn("(0.00%)", str(error), str(error))
                else:
                    raise

    def test_model_onnx_detection(self):
        model = detr_resnet50(pretrained=False).eval()
        dummy_image = torch.ones(1, 3, 800, 800) * 0.3
        model(dummy_image)

        # Test exported model on images of different size, or dummy input
        self.run_model(
            model,
            [(torch.rand(1, 3, 750, 800),)],
            input_names=["inputs"],
            output_names=["pred_logits", "pred_boxes"],
            tolerate_small_mismatch=True,
        )


class CrossBoundedBoxIouTester(unittest.TestCase):
    def test_multiple_dimensions_tight(self):
        a = torch.tensor([1, 1, 2, 2])
        b = torch.tensor([1, 2, 3, 5])
        aa = torch.cat((a, a))
        bb = torch.cat((b, b))
        ab = -0.1250
        self.assertTrue(
            torch.equal(
                box_ops.cross_bounded_box_iou_with_bounds(a[None, :], bb[None, :]),
                torch.Tensor([[ab]]),
            )
        )
        self.assertTrue(
            torch.equal(
                box_ops.cross_bounded_box_iou_with_bounds(
                    a[None, None, :], bb[None, None, :]
                ),
                torch.Tensor([[[ab]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                box_ops.cross_bounded_box_iou_with_bounds(
                    a[None, None, None, :], bb[None, None, None, :]
                ),
                torch.Tensor([[[[ab]]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                box_ops.cross_bounded_box_iou_with_bounds(
                    torch.stack([a, a, b, b]), torch.stack([aa, bb])
                ),
                torch.Tensor(torch.Tensor([[1, ab], [1, ab], [ab, 1], [ab, 1]])),
            )
        )

    def test_between_boundaries(self):
        a = torch.tensor([0, 0, 5, 5])
        b = torch.tensor([2, 2, 3, 3])
        c = torch.tensor([1, 1, 4, 4])
        ba = torch.cat((b, a))
        assert_close(
            box_ops.cross_bounded_box_iou_with_bounds(c[None, :], ba[None, :]),
            torch.Tensor([[1]]),
            rtol=0,
            atol=0,
        )

    def test_universe_target(self):
        a = torch.tensor([1, 3, 2, 2])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.tensor([0, 0, 1, 1])
        bb = torch.cat((b, a))
        self.assertTrue(
            torch.equal(
                box_ops.cross_bounded_box_iou_with_bounds(c[None, :], bb[None, :]),
                torch.Tensor([[1]]),
            )
        )

    def test_empty_pred_universe_target(self):
        a = torch.tensor([0, 0, 1, -1])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.zeros(4)
        bb = torch.cat((b, a))
        self.assertTrue(
            torch.equal(
                box_ops.cross_bounded_box_iou_with_bounds(c[None, :], bb[None, :]),
                torch.Tensor([[1]]),
            )
        )

    def test_empty_tight(self):
        # Both a and b are "missing".
        a = torch.zeros(4)
        b = c = a
        bb = torch.cat((b, a))
        self.assertTrue(
            torch.equal(
                box_ops.cross_bounded_box_iou_with_bounds(c[None, :], bb[None, :]),
                torch.Tensor([[1]]),
            )
        )

    def test_empty_pred_distinct_from_empty_target(self):
        a = b = torch.ones(4)
        c = torch.zeros(4)
        bb = torch.cat((b, a))
        assert_close(
            box_ops.cross_bounded_box_iou_with_bounds(c[None, :], bb[None, :]),
            torch.Tensor([[-1]]),
            rtol=0,
            atol=0,
        )

    def test_empty_prediction_outside_outer_boundary(self):
        a = torch.tensor([0, 0, 1, 1])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.full((4,), 2)
        bb = torch.cat((b, a))
        assert_close(
            box_ops.cross_bounded_box_iou_with_bounds(c[None, :], bb[None, :]),
            torch.Tensor([[1]]),
            rtol=0,
            atol=0,
        )

    def test_empty_prediction_outside_outer_boundary2(self):
        a = torch.tensor([0, 0, 1, 1])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.full((4,), 2 + 1 / 64)
        bb = torch.cat((b, a))
        assert_close(
            box_ops.cross_bounded_box_iou_with_bounds(c[None, :], bb[None, :]),
            torch.Tensor([[1]]),
            rtol=0,
            atol=0,
        )

    def test_empty_prediction_inside_outer_boundary(self):
        a = torch.tensor([0, 0, 2, 2])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.ones(4)
        bb = torch.cat((b, a))
        assert_close(
            box_ops.cross_bounded_box_iou_with_bounds(c[None, :], bb[None, :]),
            torch.Tensor([[1]]),
            rtol=0,
            atol=0,
        )

    def test_prediction_inside_outer_boundary(self):
        a = torch.tensor([0, 0, 3, 3])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.tensor([1, 1, 2, 2])
        bb = torch.cat((b, a))
        assert_close(
            box_ops.cross_bounded_box_iou_with_bounds(c[None, :], bb[None, :]),
            torch.Tensor([[1]]),
            rtol=0,
            atol=0,
        )

    def test_empty_prediction_outside_inner_boundary(self):
        a = torch.tensor([1, 2, 0, 5])
        b = torch.tensor([0, 0, 1, 1])
        c = torch.full((4,), 2)
        bb = torch.cat((b, a))
        assert_close(
            box_ops.cross_bounded_box_iou_with_bounds(c[None, :], bb[None, :]),
            torch.Tensor([[-1 + 1 / 4]]),
            rtol=0,
            atol=0,
        )

    def test_empty_prediction_inside_inner_boundary(self):
        a = torch.tensor([1, 2, 0, 5])
        b = torch.tensor([0, 0, 2, 2])
        c = torch.ones(4)
        bb = torch.cat((b, a))
        assert_close(
            box_ops.cross_bounded_box_iou_with_bounds(c[None, :], bb[None, :]),
            torch.Tensor([[0 - 1 + 1]]),
            rtol=0,
            atol=0,
        )

    def test_inner_boundary_inside_prediction(self):
        a = torch.tensor([1, 2, 0, 5])
        b = torch.tensor([1, 1, 2, 2])
        c = torch.tensor([0, 0, 3, 3])
        ba = torch.cat((b, a))
        assert_close(
            box_ops.cross_bounded_box_iou_with_bounds(c[None, :], ba[None, :]),
            torch.Tensor([[1]]),
            rtol=0,
            atol=0,
        )

    def test_empty_inner_boundary_outside_prediction(self):
        a = torch.tensor([1, 2, 0, 5])
        b = torch.zeros(4)
        c = torch.tensor([1, 1, 2, 2])
        bb = torch.cat((b, a))
        assert_close(
            box_ops.cross_bounded_box_iou_with_bounds(c[None, :], bb[None, :]),
            torch.Tensor([[-1 + 1 / 4]]),
            rtol=0,
            atol=0,
        )

    def test_missing_inner_universe_outer(self):
        a = torch.tensor([-np.inf, -np.inf, np.inf, np.inf])
        b = torch.tensor([np.inf, np.inf, -np.inf, -np.inf])
        c = torch.zeros(4)
        ba = torch.cat((b, a))
        self.assertTrue(
            torch.equal(
                box_ops.cross_bounded_box_iou_with_bounds(c[None, :], ba[None, :]),
                torch.Tensor([[1]]),
            )
        )


class CrossDistTester(unittest.TestCase):
    def test_multiple_dimensions_tight(self):
        a = torch.tensor([1, 1, 2, 2])
        b = torch.tensor([1, 2, 3, 5])
        aa = torch.cat((a, a))
        bb = torch.cat((b, b))
        ab = 5
        self.assertTrue(
            torch.equal(
                _cross_dist_with_bounds(a[None, :], bb[None, :], p=1),
                torch.Tensor([[ab]]),
            )
        )
        self.assertTrue(
            torch.equal(
                _cross_dist_with_bounds(a[None, None, :], bb[None, None, :], p=1),
                torch.Tensor([[[ab]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                _cross_dist_with_bounds(
                    a[None, None, None, :], bb[None, None, None, :], p=1
                ),
                torch.Tensor([[[[ab]]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                _cross_dist_with_bounds(
                    torch.stack([a, a, b, b]), torch.stack([aa, bb]), p=1
                ),
                torch.Tensor(torch.Tensor([[0, ab], [0, ab], [ab, 0], [ab, 0]])),
            )
        )

    def test_prediction_between_boundaries(self):
        a = torch.tensor([0, 0, 5, 5])
        b = torch.tensor([2, 2, 3, 3])
        c = torch.tensor([1, 1, 4, 4])
        ba = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                _cross_dist_with_bounds(c[None, :], ba[None, :], p=p),
                torch.Tensor([[0]]),
                rtol=0,
                atol=0,
            )

    def test_multiple_dimensions_tight_random(self):
        n = 2
        seq = torch.arange(0, n, dtype=torch.float32)
        all_combinations = torch.combinations(seq, r=4, with_replacement=True)
        for p in 0, 1, 2, np.inf:
            actual = _cross_dist_with_bounds(
                all_combinations,
                torch.cat((all_combinations, all_combinations), dim=-1),
                p=p,
            )
            expected = torch.cdist(all_combinations, all_combinations, p=p)
        assert_close(actual, expected)

    def test_universe_target(self):
        a = torch.tensor([1, 3, 2, 2])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.tensor([0, 0, 1, 1])
        bb = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                _cross_dist_with_bounds(c[None, :], bb[None, :], p=p),
                torch.Tensor([[0]]),
                rtol=0,
                atol=0,
            )

    def test_empty_pred_universe_target(self):
        a = torch.tensor([0, 0, 1, -1])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.zeros(4)
        bb = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                _cross_dist_with_bounds(c[None, :], bb[None, :], p=p),
                torch.Tensor([[0]]),
                rtol=0,
                atol=0,
            )

    def test_empty_tight(self):
        a = torch.zeros(4)
        b = c = a
        bb = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                _cross_dist_with_bounds(c[None, :], bb[None, :], p=p),
                torch.Tensor([[0]]),
                rtol=0,
                atol=0,
            )

    def test_empty_pred_distinct_from_empty_target(self):
        a = b = torch.ones(4)
        c = torch.zeros(4)
        ba = torch.cat((b, a))
        assert_close(
            _cross_dist_with_bounds(c[None, :], ba[None, :], p=0),
            torch.Tensor([[4]]),
            rtol=0,
            atol=0,
        )
        assert_close(
            _cross_dist_with_bounds(c[None, :], ba[None, :], p=1),
            torch.Tensor([[4]]),
            rtol=0,
            atol=0,
        )

    def test_empty_prediction_outside_outer_boundary(self):
        a = torch.tensor([0, 0, 1, 1])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.full((4,), 2)
        ba = torch.cat((b, a))
        for offset_x in range(4):
            for offset_y in range(4):
                assert_close(
                    _cross_dist_with_bounds(
                        (
                            c
                            + offset_x * torch.tensor([1, 0, 1, 0])
                            + offset_y * torch.tensor([1, 0, 1, 0])
                        )[None, :],
                        ba[None, :],
                        p=1,
                    ),
                    torch.Tensor([[4 + offset_x * 2 + offset_y * 2]]),
                    rtol=0,
                    atol=0,
                )

    def test_empty_prediction_inside_outer_boundary(self):
        a = torch.tensor([0, 0, 2, 2])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.ones(4)
        bb = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                _cross_dist_with_bounds(c[None, :], bb[None, :], p=p),
                torch.Tensor([[0]]),
                rtol=0,
                atol=0,
            )

    def test_prediction_inside_outer_boundary(self):
        a = torch.tensor([0, 0, 3, 3])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.tensor([1, 1, 2, 2])
        bb = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                _cross_dist_with_bounds(c[None, :], bb[None, :], p=p),
                torch.Tensor([[0]]),
                rtol=0,
                atol=0,
            )

    def test_empty_prediction_outside_inner_boundary(self):
        a = torch.tensor([1, 2, 0, 5])
        b = torch.tensor([0, 0, 1, 1])
        c = torch.full((4,), 2)
        ba = torch.cat((b, a))
        assert_close(
            _cross_dist_with_bounds(c[None, :], ba[None, :], p=1),
            torch.Tensor([[4]]),
            rtol=0,
            atol=0,
        )

    def test_empty_prediction_inside_inner_boundary(self):
        a = torch.tensor([1, 2, 0, 5])
        b = torch.tensor([0, 0, 2, 2])
        c = torch.ones(4)
        bb = torch.cat((b, a))
        assert_close(
            _cross_dist_with_bounds(c[None, :], bb[None, :], p=1),
            torch.Tensor([[4]]),
            rtol=0,
            atol=0,
        )

    def test_inner_boundary_inside_prediction(self):
        a = torch.tensor([1, 2, 0, 5])
        b = torch.tensor([1, 1, 2, 2])
        c = torch.tensor([0, 0, 3, 3])
        ba = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                _cross_dist_with_bounds(c[None, :], ba[None, :], p=p),
                torch.Tensor([[0]]),
                rtol=0,
                atol=0,
            )

    def test_empty_inner_boundary_outside_prediction(self):
        a = torch.tensor([1, 2, 0, 5])
        b = torch.zeros(4)
        c = torch.tensor([1, 1, 2, 2])
        ba = torch.cat((b, a))
        assert_close(
            _cross_dist_with_bounds(c[None, :], ba[None, :], p=1),
            torch.Tensor([[2]]),
            rtol=0,
            atol=0,
        )

    def test_random_tight(self):
        a = torch.rand((4,))
        c = torch.rand((4,))
        # torch.cdist does not have the "missing" convention so it gives diferent results when c is missing.
        c = make_present(c)
        cc = torch.cat((c, c))
        for p in 0, 1, 2, np.inf:
            assert_close(
                _cross_dist_with_bounds(a[None, :], cc[None, :], p=p),
                torch.cdist(a[None, :], c[None, :], p),
            )


class LpLossWithBoundsTester(unittest.TestCase):
    def test_multiple_dimensions_tight(self):
        a = torch.tensor([1, 1, 2, 2])
        b = torch.tensor([1, 2, 3, 5])
        aa = torch.cat((a, a))
        bb = torch.cat((b, b))
        ab = 5
        assert_close(
            box_ops._lp_loss_with_bounds(a, bb, p=1),
            torch.tensor(ab),
            rtol=0,
            atol=0,
            check_dtype=False,
        )
        assert_close(
            box_ops._lp_loss_with_bounds(a[None, :], bb[None, :], p=1),
            torch.Tensor([ab]),
            rtol=0,
            atol=0,
        )
        assert_close(
            box_ops._lp_loss_with_bounds(a[None, None, :], bb[None, None, :], p=1),
            torch.Tensor([[ab]]),
            rtol=0,
            atol=0,
        )
        assert_close(
            box_ops._lp_loss_with_bounds(
                a[None, None, None, :], bb[None, None, None, :], p=1
            ),
            torch.Tensor([[[ab]]]),
            rtol=0,
            atol=0,
        )
        assert_close(
            box_ops._lp_loss_with_bounds(
                torch.stack([a, a, b, b]), torch.stack([aa, bb, aa, bb]), p=1
            ),
            torch.Tensor(torch.Tensor([0, ab, ab, 0])),
            rtol=0,
            atol=0,
        )

    def test_neighbours_tight(self):
        a = torch.tensor([0, 0, 2, 2])
        b = a
        c = torch.tensor([2, 0, 4, 2])
        ba = torch.cat((b, a))
        assert_close(
            box_ops._lp_loss_with_bounds(c, ba, p=1),
            # Compare to LpLossWithBoundsCxCyWHTester.test_neighbours_tight where it's 2.
            torch.tensor(4),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_prediction_between_boundaries(self):
        a = torch.tensor([0, 0, 5, 5])
        b = torch.tensor([2, 2, 3, 3])
        c = torch.tensor([1, 1, 4, 4])
        ba = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                box_ops._lp_loss_with_bounds(c, ba, p=p),
                torch.tensor(0),
                rtol=0,
                atol=0,
                check_dtype=False,
            )

    def test_universe_target(self):
        a = torch.tensor([1, 3, 2, 2])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.tensor([0, 0, 1, 1])
        ba = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                box_ops._lp_loss_with_bounds(c, ba, p=p),
                torch.tensor(0),
                rtol=0,
                atol=0,
                check_dtype=False,
            )

    def test_empty_pred_universe_target(self):
        a = torch.tensor([0, 0, 1, -1])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.zeros(4)
        ba = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                box_ops._lp_loss_with_bounds(c, ba, p=p),
                torch.tensor(0),
                rtol=0,
                atol=0,
                check_dtype=False,
            )

    def test_empty_tight(self):
        a = torch.zeros(4)
        b = c = a
        ba = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                box_ops._lp_loss_with_bounds(c, ba, p=p),
                torch.tensor(0),
                rtol=0,
                atol=0,
                check_dtype=False,
            )

    def test_empty_pred_distinct_from_empty_target(self):
        a = b = torch.ones(4)
        c = torch.zeros(4)
        ba = torch.cat((b, a))
        assert_close(
            box_ops._lp_loss_with_bounds(c, ba, p=0),
            torch.tensor(4),
            rtol=0,
            atol=0,
            check_dtype=False,
        )
        assert_close(
            box_ops._lp_loss_with_bounds(c, ba, p=1),
            torch.tensor(4),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_empty_prediction_outside_outer_boundary(self):
        a = torch.tensor([0, 0, 1, 1])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.full((4,), 2)
        ba = torch.cat((b, a))
        for offset_x in range(4):
            for offset_y in range(4):
                actual = box_ops._lp_loss_with_bounds(
                    (
                        c
                        + offset_x * torch.tensor([1, 0, 1, 0])
                        + offset_y * torch.tensor([1, 0, 1, 0])
                    ),
                    ba,
                    p=1,
                )
                assert_close(
                    actual,
                    torch.tensor(4 + offset_x * 2 + offset_y * 2),
                    rtol=0,
                    atol=0,
                    check_dtype=False,
                )

    def test_empty_prediction_inside_outer_boundary(self):
        a = torch.tensor([0, 0, 2, 2])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.ones(4)
        bb = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                box_ops._lp_loss_with_bounds(c, bb, p=p),
                torch.tensor(0),
                rtol=0,
                atol=0,
                check_dtype=False,
            )

    def test_empty_prediction_inside_empty_outer_boundary(self):
        a = torch.tensor([9, 9, 9, 9])
        b = torch.tensor([1, 2, 0, 5])
        c = a
        bb = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                box_ops._lp_loss_with_bounds(c, bb, p=p),
                torch.tensor(0),
                rtol=0,
                atol=0,
                check_dtype=False,
            )

    def test_empty_prediction_outside_empty_outer_boundary(self):
        a = torch.tensor([9, 9, 9, 9])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.tensor([1, 2, 3, 4])
        ba = torch.cat((b, a))
        diffs = (a - c).float()

        for p in 0, 1, 2, np.inf:
            assert_close(
                box_ops._lp_loss_with_bounds(c, ba, p=p),
                LA.vector_norm(diffs, ord=p),
                rtol=0,
                atol=0,
                check_dtype=False,
            )

    def test_empty_prediction_outside_outer_boundary_with_one_empty_side(self):
        a = torch.tensor([9, -np.inf, 9, np.inf])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.tensor([1, 2, 3, 4])
        ba = torch.cat((b, a))
        diffs = (a[0::2] - c[0::2]).float()

        for p in 0, 1, 2, np.inf:
            assert_close(
                box_ops._lp_loss_with_bounds(c, ba, p=p),
                LA.vector_norm(diffs, ord=p),
                rtol=0,
                atol=0,
                check_dtype=False,
            )

    def test_prediction_inside_outer_boundary(self):
        a = torch.tensor([0, 0, 3, 3])
        b = torch.tensor([1, 2, 0, 5])
        c = torch.tensor([1, 1, 2, 2])
        bb = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                box_ops._lp_loss_with_bounds(c, bb, p=p),
                torch.tensor(0),
                rtol=0,
                atol=0,
                check_dtype=False,
            )

    def test_empty_prediction_outside_inner_boundary(self):
        a = torch.tensor([1, 2, 0, 5])
        b = torch.tensor([0, 0, 1, 1])
        c = torch.full((4,), 2)
        ba = torch.cat((b, a))
        assert_close(
            box_ops._lp_loss_with_bounds(c, ba, p=1),
            torch.tensor(4),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_empty_prediction_inside_inner_boundary(self):
        a = torch.tensor([1, 2, 0, 5])
        b = torch.tensor([0, 0, 2, 2])
        c = torch.ones(4)
        bb = torch.cat((b, a))
        assert_close(
            box_ops._lp_loss_with_bounds(c, bb, p=1),
            torch.tensor(4),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_inner_boundary_inside_prediction(self):
        a = torch.tensor([1, 2, 0, 5])
        b = torch.tensor([1, 1, 2, 2])
        c = torch.tensor([0, 0, 3, 3])
        ba = torch.cat((b, a))
        for p in 0, 1, 2, np.inf:
            assert_close(
                box_ops._lp_loss_with_bounds(c, ba, p=p),
                torch.tensor(0),
                rtol=0,
                atol=0,
                check_dtype=False,
            )

    def test_empty_inner_boundary_outside_prediction(self):
        a = torch.tensor([1, 2, 0, 5])
        b = torch.zeros(4)
        c = torch.tensor([1, 1, 2, 2])
        ba = torch.cat((b, a))
        assert_close(
            box_ops._lp_loss_with_bounds(c, ba, p=1),
            torch.tensor(2),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_random_tight(self):
        a = torch.rand((4,))
        c = torch.rand((4,))
        # torch.cdist does not have the "missing" convention so it gives diferent results when c is missing.
        c = make_present(c)
        cc = torch.cat((c, c))
        assert_close(
            box_ops._lp_loss_with_bounds(a[None, :], cc[None, :], p=1).sum(),
            nn.L1Loss(reduction="sum")(a[None, :], c[None, :]),
        )
        assert_close(
            (box_ops._lp_loss_with_bounds(a[None, :], cc[None, :], p=2) ** 2).sum(),
            nn.MSELoss(reduction="sum")(a[None, :], c[None, :]),
        )


def _smallest_distance_cxcywh(c_cxcywg, b_cxcywg, a_cxcywg, p):
    a = box_ops.box_cxcywh_to_xyxy(a_cxcywg)
    b = box_ops.box_cxcywh_to_xyxy(b_cxcywg)
    cp = torch.cartesian_prod(
        torch.arange(a[0], b[0] + 1),
        torch.arange(a[1], b[1] + 1),
        torch.arange(b[2], a[2] + 1),
        torch.arange(b[3], a[3] + 1),
    )
    cs = box_ops.box_xyxy_to_cxcywh(cp)
    # print(a.shape, b.shape, cs.shape)
    losses = LA.vector_norm(c_cxcywg[..., None, :] - cs, ord=p, dim=(-1))
    # print(losses.shape)
    return losses.min(dim=-1)


class LpLossWithBoundsCxCyWHTester(unittest.TestCase):
    def test_multiple_dimensions_tight(self):
        a = torch.tensor([1, 1, 2, 2])
        b = torch.tensor([1, 2, 3, 5])
        aa = torch.cat((a, a))
        bb = torch.cat((b, b))
        ab = 5
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(a, bb, p=1),
            torch.tensor(ab),
            rtol=0,
            atol=0,
            check_dtype=False,
        )
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(a[None, :], bb[None, :], p=1),
            torch.Tensor([ab]),
            rtol=0,
            atol=0,
        )
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(
                a[None, None, :], bb[None, None, :], p=1
            ),
            torch.Tensor([[ab]]),
            rtol=0,
            atol=0,
        )
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(
                a[None, None, None, :], bb[None, None, None, :], p=1
            ),
            torch.Tensor([[[ab]]]),
            rtol=0,
            atol=0,
        )
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(
                torch.stack([a, a, b, b]), torch.stack([aa, bb, aa, bb]), p=1
            ),
            torch.Tensor(torch.Tensor([0, ab, ab, 0])),
            rtol=0,
            atol=0,
        )

    def test_between_boundaries(self):
        a = torch.tensor([2.5, 2.5, 5, 5])
        b = torch.tensor([2.5, 2.5, 1, 1])
        c = torch.tensor([2.5, 2.5, 3, 3])
        ba = torch.cat((b, a))
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(c, ba, p=1),
            torch.tensor(0),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_neighbours_tight(self):
        a = torch.tensor([1, 1, 2, 2])
        b = a
        c = torch.tensor([3, 1, 2, 2])
        ba = torch.cat((b, a))
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(c, ba, p=1),
            # Compare to LpLossWithBoundsTester.test_neighbours_tight where it's 4.
            torch.tensor(2),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_universe_target(self):
        a = torch.tensor([1.5, 2.5, -1, 1])
        b = torch.tensor([0.5, 3.5, -1, 3])
        c = torch.tensor([0.5, 0.5, 1, 1])
        ba = torch.cat((b, a))
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(c, ba, p=1),
            torch.tensor(0),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_empty_pred_universe_target(self):
        a = torch.tensor([0.5, -0.5, 1, -1])
        b = torch.tensor([0.5, 3.5, -1, 3])
        c = torch.zeros(4)
        ba = torch.cat((b, a))
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(c, ba, p=1),
            torch.tensor(0),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_empty_tight(self):
        a = torch.zeros(4)
        b = c = a
        ba = torch.cat((b, a))
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(c, ba, p=1),
            torch.tensor(0),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_empty_pred_distinct_from_empty_target(self):
        a = b = torch.ones(4)
        c = torch.zeros(4)
        ba = torch.cat((b, a))
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(c, ba, p=1),
            torch.tensor(4),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_empty_prediction_outside_outer_boundary(self):
        a = torch.tensor([0.5, 0.5, 1, 1])
        b = torch.tensor([0.5, 3.5, -1, -3])
        c = torch.tensor([2, 2, 0, 0])
        ba = torch.cat((b, a))
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(c, ba, p=1),
            torch.tensor(2),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_empty_prediction_inside_outer_boundary(self):
        a = torch.tensor([1, 1, 2, 2])
        b = torch.tensor([0.5, 3.5, -1, -3])
        c = torch.ones(4)
        ba = torch.cat((b, a))
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(c, ba, p=1),
            torch.tensor(0),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_prediction_inside_outer_boundary(self):
        a = torch.tensor([1.5, 1.5, 3, 3])
        b = torch.tensor([0.5, 3.5, -1, 3])
        c = torch.tensor([1.5, 1.5, 1, 1])
        ba = torch.cat((b, a))
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(c, ba, p=1),
            torch.tensor(0),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_empty_prediction_outside_inner_boundary(self):
        a = torch.tensor([0.5, 3.5, -1, 3])
        b = torch.tensor([0.5, 0.5, 1, 1])
        c = torch.tensor([2, 2, 0, 0])
        ba = torch.cat((b, a))
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(c, ba, p=1),
            torch.tensor(5),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_empty_prediction_inside_inner_boundary(self):
        a = torch.tensor([0.5, 3.5, -1, 3])
        b = torch.tensor([1, 1, 2, 2])
        c = torch.ones(4)
        ba = torch.cat((b, a))
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(c, ba, p=1),
            torch.tensor(2),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_inner_boundary_inside_prediction(self):
        a = torch.tensor([0.5, 3.5, -1, 3])
        b = torch.tensor([1.5, 1.5, 1, 1])
        c = torch.tensor([1.5, 1.5, 3, 3])
        ba = torch.cat((b, a))
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(c, ba, p=1),
            torch.tensor(0),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_empty_inner_boundary_outside_prediction(self):
        a = torch.tensor([0.5, 3.5, -1, -3])
        b = torch.zeros(4)
        c = torch.tensor([1.5, 1.5, 1, 1])
        ba = torch.cat((b, a))
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(c, ba, p=1),
            torch.tensor(2),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_random_tight(self):
        a = torch.rand((4,))
        c = torch.rand((4,))
        # torch.cdist does not have the "missing" convention so it gives diferent results when c is missing.
        c = make_present(c)
        cc = torch.cat((c, c))
        assert_close(
            box_ops.lp_loss_with_bounds_cxcywh(a[None, :], cc[None, :], p=1).sum(),
            nn.L1Loss(reduction="sum")(a[None, :], c[None, :]),
        )
        assert_close(
            (
                box_ops.lp_loss_with_bounds_cxcywh(a[None, :], cc[None, :], p=2) ** 2
            ).sum(),
            nn.MSELoss(reduction="sum")(a[None, :], c[None, :]),
        )

    def test_comprehensive(self):
        # Helps so as to deal only with ints. For int inputs there is always a
        # nearest mathing rectangle with all values multiples of 0.5, so this
        # factor will ensure there is always a nearest matching rectangle with
        # integral center and length values.
        #
        # torch.manual_seed(0)
        factor = 2
        l = 4
        seq = torch.arange(0, 3 * l) * factor
        preds = torch.combinations(seq, r=4, with_replacement=True)
        for _ in range(3):
            r = torch.randint(l, l << 1, (4,))
            a = r[:4]
            b_center = torch.cat(
                [
                    torch.randint(r[0] - r[2] // 2, r[0] + r[2] // 2 + 1, (1,)),
                    torch.randint(r[1] - r[3] // 2, r[1] + r[3] // 2 + 1, (1,)),
                ]
            )
            max_wh = r[2:] - 2 * (b_center - r[:2]).abs()
            b = torch.cat(
                [
                    b_center,
                    torch.randint(0, max_wh[0] + 1, (1,)),
                    torch.randint(0, max_wh[1] + 1, (1,)),
                ]
            )
            a, b = (torch.stack([a, b]) * factor).unbind()
            ba = torch.cat([b, a], dim=-1)
            for p in [1]:
                for c in preds:
                    expected, _ = _smallest_distance_cxcywh(c, b, a, p=p)
                    actual = box_ops.lp_loss_with_bounds_cxcywh(c, ba, p=p)
                    # assert_close(actual, expected, rtol=0, atol=0)
                    assert_close(actual, expected)


class CrossLpLossWithBoundsCxCyWHTester(unittest.TestCase):
    def test_comprehensive(self):
        # Helps so as to deal only with ints. For int inputs there is always a
        # nearest mathing rectangle with all values multiples of 0.5, so this
        # factor will ensure there is always a nearest mathing rectangle with
        # integral center and length values.
        #
        # torch.manual_seed(0)
        factor = 2
        l = 5
        seq = torch.arange(0, 3 * l) * factor
        pred = torch.combinations(seq, r=4, with_replacement=True)
        bas = []
        for _ in range(5):
            r = torch.randint(l, l << 1, (4,))
            a = r[:4]
            b_center = torch.cat(
                [
                    torch.randint(r[0] - r[2] // 2, r[0] + r[2] // 2 + 1, (1,)),
                    torch.randint(r[1] - r[3] // 2, r[1] + r[3] // 2 + 1, (1,)),
                ]
            )
            max_wh = r[2:] - 2 * (b_center - r[:2]).abs()
            b = torch.cat(
                [
                    b_center,
                    torch.randint(0, max_wh[0] + 1, (1,)),
                    torch.randint(0, max_wh[1] + 1, (1,)),
                ]
            )
            a, b = (torch.stack([a, b]) * factor).unbind()
            ba = torch.cat([b, a], dim=-1)
            bas.append(ba)
        actual = box_ops.cross_lp_loss_with_bounds_cxcywh(pred, torch.stack(bas), p=1)
        for actual_row, ba in zip(actual.transpose(0, 1), bas, strict=True):
            expected, _ = _smallest_distance_cxcywh(pred, ba[:4], ba[4:], p=1)
            assert_close(actual_row, expected)


class LinearCoefficientsTester(unittest.TestCase):
    def test_interval_bounds(self):
        # linear coefficients
        lc_dist = Uniform(-1, 1)
        m = 100
        lc = lc_dist.sample(sample_shape=(m, 4))
        # print(lc)

        random_theta, random_phi, random_alpha, random_beta = torch.unbind(lc, dim=1)
        theta_alpha_condition = random_theta.abs() >= random_alpha.abs()
        theta = random_theta.where(theta_alpha_condition, random_alpha)
        alpha = random_alpha.where(theta_alpha_condition, random_theta)
        del random_theta, random_alpha

        phi_beta_condition = random_phi.abs() >= random_beta.abs()
        phi = random_phi.where(phi_beta_condition, random_beta)
        beta = random_beta.where(phi_beta_condition, random_phi)
        del random_phi, random_beta

        # Ensure
        # alpha.abs() <= theta.abs()
        # beta.abs() <= phi.abs()

        ab_dist = Uniform(-4, 4)
        n = 50
        ab = ab_dist.sample(sample_shape=(n, 2))

        a, b = torch.unbind(ab, dim=1)

        expanded_theta = theta[:, None].expand(m, n)
        expanded_phi = phi[:, None].expand(m, n)
        expanded_alpha = alpha[:, None].expand(m, n)
        expanded_beta = beta[:, None].expand(m, n)
        expanded_a = a[None, :].expand(m, n)
        expanded_b = b[None, :].expand(m, n)

        sign = (
            expanded_a.sign()
            * expanded_theta.sign()
            * expanded_b.sign()
            * expanded_phi.sign()
        )

        def check_inequality(x, y, cond):
            left = (expanded_theta[cond] * x + expanded_phi[cond] * y).abs()
            right = (expanded_alpha[cond] * x + expanded_beta[cond] * y).abs()
            assert_close(left.min(right), right, rtol=0, atol=0)

        condition = sign >= 0
        x, y = expanded_a[condition], expanded_b[condition]
        check_inequality(x, y, condition)
        del condition

        condition = sign <= 0
        check_inequality(
            torch.zeros(condition.sum()),
            expanded_a[condition] * expanded_theta[condition]
            + expanded_b[condition] * expanded_phi[condition],
            condition,
        )
        check_inequality(
            expanded_a[condition] * expanded_theta[condition]
            + expanded_b[condition] * expanded_phi[condition],
            torch.zeros(condition.sum()),
            condition,
        )
        del condition


class LinearCoefficientsCenterSizeTester(unittest.TestCase):
    def test_prediction_between_boundaries(self):
        o = torch.tensor([0, 5])
        h = torch.tensor([2, 3])
        alpha, beta = 0.5, 0.5
        theta, phi = -1, 1
        hole = torch.tensor([alpha * h[0] + beta * h[1], theta * h[0] + phi * h[1]])
        outer = torch.tensor([alpha * o[0] + beta * o[1], theta * o[0] + phi * o[1]])
        for b0 in torch.arange(o[0], h[0] + 1):
            for b1 in torch.arange(h[1], o[1] + 1):
                b = torch.tensor([b0, b1])
                pred = torch.tensor(
                    [alpha * b[0] + beta * b[1], theta * b[0] + phi * b[1]]
                )
                target = torch.cat((hole, outer))
                m, n = 2, 3
                (
                    phi_alpha_minus_theta_beta,
                    s,
                    factor,
                    x,
                ) = box_ops.cross_l1_loss_linear_neighbour_unscaled(
                    pred[None, :].expand(m, 2),
                    target[None, :].expand(n, 4),
                    alpha=alpha,
                    beta=beta,
                    theta=theta,
                    phi=phi,
                )
                self.assertEqual(1, phi_alpha_minus_theta_beta)
                self.assertEqual(-1, factor)
                assert_close(
                    s,
                    pred[1][None, None, None].expand(m, n, 1),
                    rtol=0,
                    atol=0,
                    check_dtype=False,
                )
                assert_close(
                    x,
                    factor * pred[None, None, 0, None].expand(m, n, 1),
                    rtol=0,
                    atol=0,
                )
                assert_close(
                    box_ops.cross_l1_loss_with_bounds_linear(
                        pred[None, :].expand(m, 2),
                        target[None, :].expand(n, 4),
                        alpha=alpha,
                        beta=beta,
                        theta=theta,
                        phi=phi,
                    ),
                    torch.zeros(m, n),
                    rtol=0,
                    atol=0,
                )

    def test_prediction_too_short(self):
        o = torch.tensor([0, 6])
        h = torch.tensor([2, 4])
        alpha, beta = 0.5, 0.5
        theta, phi = -1, 1
        hole = torch.tensor([alpha * h[0] + beta * h[1], theta * h[0] + phi * h[1]])
        outer = torch.tensor([alpha * o[0] + beta * o[1], theta * o[0] + phi * o[1]])
        b = torch.tensor([3, 3])
        pred = torch.tensor([alpha * b[0] + beta * b[1], theta * b[0] + phi * b[1]])
        target = torch.cat((hole, outer))
        m, n = 1, 1
        (
            phi_alpha_minus_theta_beta,
            s,
            factor,
            x,
        ) = box_ops.cross_l1_loss_linear_neighbour_unscaled(
            pred[None, :].expand(m, 2),
            target[None, :].expand(n, 4),
            alpha=alpha,
            beta=beta,
            theta=theta,
            phi=phi,
        )
        self.assertEqual(1, phi_alpha_minus_theta_beta)
        self.assertEqual(-1, factor)
        assert_close(
            s,
            (h[1] - h[0])[None, None, None].expand(m, n, 1),
            rtol=0,
            atol=0,
            check_dtype=False,
        )
        assert_close(
            x,
            factor * torch.tensor(3)[None, None, None].expand(m, n, 1),
            rtol=0,
            atol=0,
        )
        actual_distance = box_ops.cross_l1_loss_with_bounds_linear(
            pred[None, :].expand(m, 2),
            target[None, :].expand(n, 4),
            alpha=alpha,
            beta=beta,
            theta=theta,
            phi=phi,
        )
        assert_close(
            actual_distance,
            torch.full((m, n), 2),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_prediction_too_long(self):
        o = torch.tensor([0, 6])
        h = torch.tensor([2, 4])
        alpha, beta = 0.5, 0.5
        theta, phi = -1, 1
        hole = torch.tensor([alpha * h[0] + beta * h[1], theta * h[0] + phi * h[1]])
        outer = torch.tensor([alpha * o[0] + beta * o[1], theta * o[0] + phi * o[1]])
        b = torch.tensor([-1, 7])
        pred = torch.tensor([alpha * b[0] + beta * b[1], theta * b[0] + phi * b[1]])
        target = torch.cat((hole, outer))
        m, n = 1, 1
        (
            phi_alpha_minus_theta_beta,
            s,
            factor,
            x,
        ) = box_ops.cross_l1_loss_linear_neighbour_unscaled(
            pred[None, :].expand(m, 2),
            target[None, :].expand(n, 4),
            alpha=alpha,
            beta=beta,
            theta=theta,
            phi=phi,
        )
        self.assertEqual(1, phi_alpha_minus_theta_beta)
        self.assertEqual(-1, factor)
        assert_close(
            s,
            (o[1] - o[0])[None, None, None].expand(m, n, 1),
            rtol=0,
            atol=0,
            check_dtype=False,
        )
        assert_close(
            x,
            factor * torch.tensor(3)[None, None, None].expand(m, n, 1),
            rtol=0,
            atol=0,
        )
        actual_distance = box_ops.cross_l1_loss_with_bounds_linear(
            pred[None, :].expand(m, 2),
            target[None, :].expand(n, 4),
            alpha=alpha,
            beta=beta,
            theta=theta,
            phi=phi,
        )
        assert_close(
            actual_distance,
            torch.full((m, n), 2),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_prediction_too_far_to_the_left(self):
        o = torch.tensor([0, 6])
        h = torch.tensor([2, 4])
        alpha, beta = 0.5, 0.5
        theta, phi = -1, 1
        hole = torch.tensor([alpha * h[0] + beta * h[1], theta * h[0] + phi * h[1]])
        outer = torch.tensor([alpha * o[0] + beta * o[1], theta * o[0] + phi * o[1]])
        b = torch.tensor([-5, -1])
        pred = torch.tensor([alpha * b[0] + beta * b[1], theta * b[0] + phi * b[1]])
        target = torch.cat((hole, outer))
        m, n = 1, 1
        (
            phi_alpha_minus_theta_beta,
            s,
            factor,
            x,
        ) = box_ops.cross_l1_loss_linear_neighbour_unscaled(
            pred[None, :].expand(m, 2),
            target[None, :].expand(n, 4),
            alpha=alpha,
            beta=beta,
            theta=theta,
            phi=phi,
        )
        self.assertEqual(1, phi_alpha_minus_theta_beta)
        self.assertEqual(-1, factor)
        assert_close(
            s,
            (b[1] - b[0])[None, None, None].expand(m, n, 1),
            rtol=0,
            atol=0,
            check_dtype=False,
        )
        assert_close(
            x,
            factor * torch.tensor(2)[None, None, None].expand(m, n, 1),
            rtol=0,
            atol=0,
        )
        actual_distance = box_ops.cross_l1_loss_with_bounds_linear(
            pred[None, :].expand(m, 2),
            target[None, :].expand(n, 4),
            alpha=alpha,
            beta=beta,
            theta=theta,
            phi=phi,
        )
        assert_close(
            actual_distance,
            torch.full((m, n), 5),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_prediction_too_far_to_the_right(self):
        o = torch.tensor([0, 6])
        h = torch.tensor([2, 4])
        alpha, beta = 0.5, 0.5
        theta, phi = -1, 1
        hole = torch.tensor([alpha * h[0] + beta * h[1], theta * h[0] + phi * h[1]])
        outer = torch.tensor([alpha * o[0] + beta * o[1], theta * o[0] + phi * o[1]])
        b = torch.tensor([7, 11])
        pred = torch.tensor([alpha * b[0] + beta * b[1], theta * b[0] + phi * b[1]])
        target = torch.cat((hole, outer))
        m, n = 1, 1
        (
            phi_alpha_minus_theta_beta,
            s,
            factor,
            x,
        ) = box_ops.cross_l1_loss_linear_neighbour_unscaled(
            pred[None, :].expand(m, 2),
            target[None, :].expand(n, 4),
            alpha=alpha,
            beta=beta,
            theta=theta,
            phi=phi,
        )
        self.assertEqual(1, phi_alpha_minus_theta_beta)
        self.assertEqual(-1, factor)
        assert_close(
            s,
            (b[1] - b[0])[None, None, None].expand(m, n, 1),
            rtol=0,
            atol=0,
            check_dtype=False,
        )
        assert_close(
            x,
            factor * torch.tensor(4)[None, None, None].expand(m, n, 1),
            rtol=0,
            atol=0,
        )
        actual_distance = box_ops.cross_l1_loss_with_bounds_linear(
            pred[None, :].expand(m, 2),
            target[None, :].expand(n, 4),
            alpha=alpha,
            beta=beta,
            theta=theta,
            phi=phi,
        )
        assert_close(
            actual_distance,
            torch.full((m, n), 5),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_prediction_dim0_too_far_to_the_right_dim1_too_long_same_target(self):
        o = torch.tensor([0, 6])
        h = torch.tensor([2, 4])
        alpha, beta = 0.5, 0.5
        theta, phi = -1, 1
        hole = torch.tensor([alpha * h[0] + beta * h[1], theta * h[0] + phi * h[1]])
        outer = torch.tensor([alpha * o[0] + beta * o[1], theta * o[0] + phi * o[1]])

        b0 = torch.tensor([7, 11])
        b1 = torch.tensor([-1, 7])

        # torch.stack((b0, b1), dim=1).view(8)

        pred0 = torch.tensor(
            [alpha * b0[0] + beta * b0[1], theta * b0[0] + phi * b0[1]]
        )
        pred1 = torch.tensor(
            [alpha * b1[0] + beta * b1[1], theta * b1[0] + phi * b1[1]]
        )
        pred = torch.stack((pred0, pred1), dim=1).view(4)

        target = torch.cat(
            (
                torch.stack((hole, hole), dim=1).view(4),
                torch.stack((outer, outer), dim=1).view(4),
            )
        )
        m, n = 1, 1
        (
            phi_alpha_minus_theta_beta,
            s,
            factor,
            x,
        ) = box_ops.cross_l1_loss_linear_neighbour_unscaled(
            pred[None, :].expand(m, -1),
            target[None, :].expand(n, -1),
            alpha=alpha,
            beta=beta,
            theta=theta,
            phi=phi,
        )
        self.assertEqual(1, phi_alpha_minus_theta_beta)
        self.assertEqual(-1, factor)
        assert_close(
            s,
            torch.stack((b0[1] - b0[0], o[1] - o[0]))[None, None, :].expand(m, n, -1),
            rtol=0,
            atol=0,
            check_dtype=False,
        )
        assert_close(
            x,
            factor * torch.tensor((4, 3))[None, None, :].expand(m, n, -1),
            rtol=0,
            atol=0,
        )
        actual_distance = box_ops.cross_l1_loss_with_bounds_linear(
            pred[None, :].expand(m, -1),
            target[None, :].expand(n, -1),
            alpha=alpha,
            beta=beta,
            theta=theta,
            phi=phi,
        )
        assert_close(
            actual_distance,
            torch.full((m, n), 7),
            rtol=0,
            atol=0,
            check_dtype=False,
        )

    def test_is_present_linear(self):
        alpha, beta = 0.5, 0.5
        theta, phi = -1, 1
        n = 1000
        boundaries = torch.randint(low=0, high=100, size=(n, 4))
        expected = box_ops.is_present_cxcywh(boundaries)
        actual = box_ops.is_present_linear(
            boundaries, alpha=alpha, beta=beta, theta=theta, phi=phi
        )
        assert_close(actual.all(dim=-1), expected, rtol=0, atol=0)

    def test_universe(self):
        torch.manual_seed(5)
        alpha, beta = 0.5, 0.5
        theta, phi = -1, 1
        preds = torch.tensor([[3, 4]])
        targets = torch.tensor([[10, -1, 100, -2]])

        self.assertTrue(
            ~box_ops.is_present_linear(
                targets, alpha=alpha, beta=beta, theta=theta, phi=phi
            ).all()
        )

        actual = box_ops.cross_l1_loss_linear_neighbour(
            preds, targets, alpha=alpha, beta=beta, theta=theta, phi=phi
        )

        assert_close(actual, preds[..., None, :], rtol=0, atol=0, check_dtype=False)

    def test_random_fully_specified(self):
        torch.manual_seed(5)
        alpha, beta = 0.5, 0.5
        theta, phi = -1, 1
        m = 100
        preds = torch.randint(low=0, high=100, size=(m, 4))
        n = 1000
        targets = torch.randint(low=0, high=100, size=(n, 4), dtype=torch.float32)

        # Make hole length smaller than outer length.
        targets[..., 1::2] = torch.cat(
            (
                targets[..., 1::2].amin(dim=-1, keepdim=True),
                targets[..., 1::2].amax(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        targets[..., 0] = targets[..., 0].clamp(
            targets[..., 2] - (targets[..., 3] - targets[..., 1]) / 2,
            targets[..., 2] + (targets[..., 3] - targets[..., 1]) / 2,
        )
        self.assertTrue(
            box_ops.is_present_linear(
                targets[..., :2], alpha=alpha, beta=beta, theta=theta, phi=phi
            ).all()
        )
        self.assertTrue(
            box_ops.is_present_linear(
                targets[..., 2:], alpha=alpha, beta=beta, theta=theta, phi=phi
            ).all()
        )

        # Due to the cxcy code always needing two dimensions, just make them identical.
        targets = targets.repeat_interleave(2, dim=-1)

        actual = box_ops.cross_l1_loss_linear_neighbour(
            preds, targets, alpha=alpha, beta=beta, theta=theta, phi=phi
        )
        expected = box_ops.cross_lp_loss_with_bounds_cxcywh_neighbour(preds, targets)

        assert_close(actual, expected, rtol=0, atol=0)

    def test_random_outer_border(self):
        torch.manual_seed(5)
        alpha, beta = 0.5, 0.5
        theta, phi = -1, 1
        m = 100
        preds = torch.randint(low=0, high=100, size=(m, 4))
        n = 1000
        targets = torch.randint(low=0, high=100, size=(n, 4), dtype=torch.float32)

        # Make the hole length negative.
        targets[..., 1] = -1 - targets[..., 1]
        self.assertFalse(
            box_ops.is_present_linear(
                targets[..., :2], alpha=alpha, beta=beta, theta=theta, phi=phi
            ).any()
        )
        self.assertTrue(
            box_ops.is_present_linear(
                targets[..., 2:], alpha=alpha, beta=beta, theta=theta, phi=phi
            ).all()
        )

        # Due to the cxcy code always needing two dimensions, just make them identical.
        targets = targets.repeat_interleave(2, dim=-1)

        actual = box_ops.cross_l1_loss_linear_neighbour(
            preds, targets, alpha=alpha, beta=beta, theta=theta, phi=phi
        )
        expected = box_ops.cross_lp_loss_with_bounds_cxcywh_neighbour(preds, targets)

        assert_close(actual, expected, rtol=0, atol=0)

    def test_random_hole_border(self):
        torch.manual_seed(5)
        alpha, beta = 0.5, 0.5
        theta, phi = -1, 1
        m = 100
        preds = torch.randint(low=0, high=100, size=(m, 4))
        n = 1000
        targets = torch.randint(low=0, high=100, size=(n, 4), dtype=torch.float32)

        # Make the outer length negative.
        targets[..., 3] = -1 - targets[..., 1]
        self.assertTrue(
            box_ops.is_present_linear(
                targets[..., :2], alpha=alpha, beta=beta, theta=theta, phi=phi
            ).all()
        )
        self.assertFalse(
            box_ops.is_present_linear(
                targets[..., 2:], alpha=alpha, beta=beta, theta=theta, phi=phi
            ).any()
        )

        # Due to the cxcy code always needing two dimensions, just make them identical.
        targets = targets.repeat_interleave(2, dim=-1)

        actual = box_ops.cross_l1_loss_linear_neighbour(
            preds, targets, alpha=alpha, beta=beta, theta=theta, phi=phi
        )
        expected = box_ops.cross_lp_loss_with_bounds_cxcywh_neighbour(preds, targets)

        assert_close(actual, expected, rtol=0, atol=0)


class LinearCoefficientsTwoThreeFourFiveTester(unittest.TestCase):
    def test_prediction_between_boundaries(self):
        o = torch.tensor([0, 5])
        h = torch.tensor([2, 3])
        alpha, beta = 2, 3
        theta, phi = 4, 5
        delta = phi * alpha - theta * beta
        hole = torch.tensor([alpha * h[0] + beta * h[1], theta * h[0] + phi * h[1]])
        outer = torch.tensor([alpha * o[0] + beta * o[1], theta * o[0] + phi * o[1]])
        for b0 in torch.arange(o[0], h[0] + 1):
            for b1 in torch.arange(h[1], o[1] + 1):
                b = torch.tensor([b0, b1])
                pred = torch.tensor(
                    [alpha * b[0] + beta * b[1], theta * b[0] + phi * b[1]]
                )
                target = torch.cat((hole, outer))
                m, n = 1, 1
                (
                    phi_alpha_minus_theta_beta,
                    s,
                    factor,
                    x,
                ) = box_ops.cross_l1_loss_linear_neighbour_unscaled(
                    pred[None, :].expand(m, 2),
                    target[None, :].expand(n, 4),
                    alpha=alpha,
                    beta=beta,
                    theta=theta,
                    phi=phi,
                )
                self.assertEqual(-2, phi_alpha_minus_theta_beta)
                self.assertEqual(-40, factor)
                assert_close(
                    s,
                    delta * pred[1][None, None, None].expand(m, n, 1),
                    rtol=0,
                    atol=0,
                    check_dtype=False,
                )
                assert_close(
                    x,
                    factor * pred[None, None, 0, None].expand(m, n, 1),
                    rtol=0,
                    atol=0,
                    check_dtype=False,
                )
                assert_close(
                    box_ops.cross_l1_loss_with_bounds_linear(
                        pred[None, :].expand(m, 2),
                        target[None, :].expand(n, 4),
                        alpha=alpha,
                        beta=beta,
                        theta=theta,
                        phi=phi,
                    ),
                    torch.zeros(m, n),
                    rtol=0,
                    atol=0,
                )


def _filter_linear_coefficients(coefficients):
    alpha, beta, theta, phi = coefficients.unbind(-1)
    delta = phi * alpha - theta * beta
    return coefficients[(alpha != 0).logical_and(theta != 0).logical_and(delta != 0)]


def _repair_linear_coefficients(coefficients):
    coefficients = _filter_linear_coefficients(coefficients)
    alpha, beta, theta, phi = coefficients.unbind(-1)
    coefficients[:, 0::2] = coefficients[:, 0::2].where(
        (theta.abs() >= alpha.abs())[:, None], coefficients[:, ::2].flip(dims=(-1,))
    )
    coefficients[:, 1::2] = coefficients[:, 1::2].where(
        (phi.abs() >= beta.abs())[:, None], coefficients[:, 1::2].flip(dims=(-1,))
    )
    alpha, beta, theta, phi = coefficients.unbind(-1)
    delta = phi * alpha - theta * beta
    return coefficients[(alpha != 0).logical_and(theta != 0).logical_and(delta != 0), :]


def _repair_target(target):
    x = target.sort().values
    return torch.stack((x[:, 1], x[:, 2], x[:, 0], x[:, 3]), dim=-1)


def _repair_pred(pred):
    return pred.sort().values


class LinearCoefficientsRandomTester(unittest.TestCase):
    def test_prediction_between_boundaries(self):
        torch.manual_seed(5)
        m = 64
        coefficients = _repair_linear_coefficients(torch.randint(-4, 5, (m, 4))).unique(
            dim=0
        )
        alpha, beta, theta, phi = coefficients.unbind(-1)

        n = 32
        target = _repair_target(torch.randint(-20, 21, (n, 4))).unique(dim=0)

        k = 16
        fractions = torch.rand((k, 2)).unique(dim=0)

        l_h, u_h, l_o, u_o = target.unbind(-1)
        l = l_o[:, None] + (l_h - l_o)[:, None] * fractions[None, :, 0]
        u = u_h[:, None] + (u_o - u_h)[:, None] * fractions[None, :, 1]

        for coefficients_row in coefficients:
            # print("coefficients_row: {}".format(coefficients_row))
            alpha, beta, theta, phi = coefficients_row.unbind(-1)
            linear_target = torch.stack(
                (
                    alpha * l_h + beta * u_h,
                    theta * l_h + phi * u_h,
                    alpha * l_o + beta * u_o,
                    theta * l_o + phi * u_o,
                ),
                dim=-1,
            )
            # print("linear_target: {}".format(linear_target))
            linear_pred = torch.stack(
                (alpha * l + beta * u, theta * l + phi * u), dim=-1
            )
            i = 0
            for linear_target_row in linear_target:
                linear_pred_row = linear_pred[i, :]
                actual = box_ops.cross_l1_loss_linear_neighbour(
                    linear_pred_row,
                    linear_target_row[None, :],
                    alpha=alpha,
                    beta=beta,
                    theta=theta,
                    phi=phi,
                )
                assert_close(actual, linear_pred_row[:, None, :])
                i += 1

    def test_comprehensive(self):
        torch.manual_seed(5)
        m = 32
        coefficients = _repair_linear_coefficients(torch.randint(-4, 5, (m, 4))).unique(
            dim=0
        )
        alpha, beta, theta, phi = coefficients.unbind(-1)

        low, high = -20, 21
        n = 8
        target = _repair_target(torch.randint(low, high, (n, 4))).unique(dim=0)

        k = 1024
        fractions = torch.rand((k, 2)).unique(dim=0)

        l_h, u_h, l_o, u_o = target.unbind(-1)
        l = l_o[:, None] + (l_h - l_o)[:, None] * fractions[None, :, 0]
        u = u_h[:, None] + (u_o - u_h)[:, None] * fractions[None, :, 1]

        q = 32
        pred = _repair_pred(torch.randint(low, high, (q, 2))).unique(dim=0)

        for coefficients_row in coefficients:
            alpha, beta, theta, phi = coefficients_row.unbind(-1)
            linear_target = torch.stack(
                (
                    alpha * l_h + beta * u_h,
                    theta * l_h + phi * u_h,
                    alpha * l_o + beta * u_o,
                    theta * l_o + phi * u_o,
                ),
                dim=-1,
            )
            grid_linear = torch.stack(
                (alpha * l + beta * u, theta * l + phi * u), dim=-1
            )
            i = 0
            for linear_target_row in linear_target:
                linear_pred = torch.stack(
                    (
                        alpha * pred[:, 0] + beta * pred[:, 1],
                        theta * pred[:, 0] + phi * pred[:, 1],
                    ),
                    dim=-1,
                )
                actual = box_ops.cross_l1_loss_with_bounds_linear(
                    linear_pred,
                    linear_target_row[None, :],
                    alpha=alpha,
                    beta=beta,
                    theta=theta,
                    phi=phi,
                )
                diff = grid_linear[i][None, :, :] - linear_pred[:, None, :]
                grid_distances = LA.vector_norm(diff, ord=1, dim=-1)
                min_grid_distances = grid_distances.min(dim=-1, keepdim=True).values
                # print("linear_neighbour_unscaled: {}".format(pprint.pformat(locals())))
                assert_close(actual.min(min_grid_distances), actual)
                i += 1


if __name__ == "__main__":
    unittest.main()
