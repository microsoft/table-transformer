# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
from torch import linalg as LA
import numpy as np
import pprint

# Arbitrarily define a canonical missing box.
MISSING_BOX = torch.tensor([5, 5, 4, 5], dtype=torch.float32)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy_with_bounds(x):
    a = box_cxcywh_to_xyxy(x[..., :4])
    b = box_cxcywh_to_xyxy(x[..., 4:])
    result = torch.cat([a, b], dim=-1)
    # print("box_cxcywh_to_xyxy_with_bounds({}: {}".format(x, result))
    return result


def inf_safe_avg(x0, x1):
    # Treat inf == -inf.
    return torch.where(x0 == -x1, 0, (x0 + x1) / 2)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [inf_safe_avg(x0, x1), inf_safe_avg(y0, y1), x1 - x0, y1 - y0]
    result = torch.stack(b, dim=-1)
    # print("box_xyxy_to_cxcywh({}): {}".format(x, result))
    return result


def box_xyxy_to_cxcywh_with_bounds(x):
    # TODO: Add unit tests.
    assert x.shape[-1] == 8
    a = box_xyxy_to_cxcywh(x[..., :4])
    b = box_xyxy_to_cxcywh(x[..., 4:])
    result = torch.cat([a, b], dim=-1)
    # print("box_xyxy_to_cxcywh_with_bounds({}): {}".format(x, result))
    return result


def multidimensional_box_area(boxes):
    return box_area(boxes.view(-1, 4)).view(boxes.shape[:-1])


def _cross_box_inter_union_with_known_area(boxes_and_areas1, boxes2):
    # print("boxes_and_areas1: {} boxes2: {}".format(boxes_and_areas1, boxes2))
    boxes1, area1 = boxes_and_areas1
    area2 = multidimensional_box_area(boxes2)

    lt = torch.max(boxes1[..., None, :2], boxes2[..., None, :, :2])  # [..., N,M,2]
    rb = torch.min(boxes1[..., None, 2:], boxes2[..., None, :, 2:])  # [..., N,M,2]

    wh = (rb - lt).clamp(min=0)  # [..., N,M,2]
    inter = wh[..., 0] * wh[..., 1]  # [..., N,M]

    union = area1[..., None] + area2[..., None, :] - inter

    # print("rb: {} lt: {}, rb-lt: {} area2: {} inter: {} union: {}".format(rb, lt, rb - lt, area2, inter, union))
    return area2, inter, union


# modified from torchvision to also return the union and to accept an arbitrary prefix of equal (or broadcastable) dimensions.
def cross_box_inter_union(boxes1, boxes2):
    area1 = multidimensional_box_area(boxes1)
    return (area1,) + _cross_box_inter_union_with_known_area((boxes1, area1), boxes2)


# modified from torchvision to also return the union and to accept an arbitrary prefix of equal (or broadcastable) dimensions.
def cross_box_iou(boxes1, boxes2):
    _, _, inter, union = cross_box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou, union


def _cross_enclosing_box_area(boxes1, boxes2):
    lt = torch.min(boxes1[..., None, :2], boxes2[..., None, :, :2])
    rb = torch.max(boxes1[..., None, 2:], boxes2[..., None, :, 2:])

    # wh = (rb - lt).clamp(min=0)  # [..., N,M,2]
    wh = rb - lt  # [..., N,M,2]
    # print("wh: {}".format(wh))
    return wh[..., 0] * wh[..., 1]


# https://github.com/pytorch/vision/blob/main/torchvision/ops/giou_loss.py
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # assert (boxes1[..., 2:] >= boxes1[..., :2]).all()
    # assert (boxes2[..., 2:] >= boxes2[..., :2]).all()
    assert boxes1.shape[-1] == 4
    assert boxes2.shape[-1] == 4
    assert len(boxes1.shape) == len(boxes2.shape)
    iou, union = cross_box_iou(boxes1, boxes2)
    area = _cross_enclosing_box_area(boxes1, boxes2)
    return iou - (area - union) / area


def expand_pred(pred, target_shape):
    return pred[..., None, :].expand(
        (-1,) * (len(pred.shape) - 1) + (target_shape[-2], -1)
    )


def is_present(boxes):
    return (boxes[..., :2] <= boxes[..., 2:]).all(dim=-1)


def is_present_cxcywh(boxes):
    assert boxes.shape[-1] == 4
    return (boxes[..., 2:] >= 0).all(dim=-1)


def is_present_and_empty(boxes):
    return is_present(boxes) & (boxes[..., :2] == boxes[..., 2:]).any(dim=-1)


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def subtract_zero_when_equal(x, y):
    diff = x - y
    return torch.where(x == y, diff.new_zeros(1), diff)


def expand_missing_outer_boundary(a):
    return torch.where(
        is_present(a)[..., None], a, a.new_tensor([-np.inf, -np.inf, np.inf, np.inf])
    )


def expand_missing_outer_boundary_cxcywh(a):
    any_finite_value_x, any_finite_value_y = 0, 0
    return torch.where(
        is_present_cxcywh(a)[..., None],
        a,
        a.new_tensor([any_finite_value_x, any_finite_value_y, np.inf, np.inf]),
    )


def shrink_missing_inner_boundary_given_outer_boundary(b, a):
    return torch.where(
        is_present(b)[..., None],
        b,
        torch.cat(
            [
                a[..., 2, None],
                a[..., 3, None],
                a[..., 0, None],
                a[..., 1, None],
            ],
            dim=-1,
        ),
    )


def shrink_missing_inner_boundary_given_outer_boundary_cxcywh(b, a):
    return torch.where(
        is_present_cxcywh(b)[..., None],
        b,
        torch.cat(
            [
                a[..., 0, None],
                a[..., 1, None],
                -a[..., 2, None],
                -a[..., 3, None],
            ],
            dim=-1,
        ),
    )


# Like torch.cdist(pred, target, p), but `target` has the format inner_boundary, outer_boundary.
# https://github.com/droyed/eucl_dist/blob/master/eucl_dist/gpu_dist.py
# https://github.com/droyed/eucl_dist/wiki/Main-Article#prospective-method
# https://stackoverflow.com/questions/52030458/vectorized-spatial-distance-in-python-using-numpy
# https://github.com/scipy/scipy/blob/b944bac904a2edb78556de3c8c5680dc78f1d461/scipy/spatial/src/distance_impl.h#L651
def cross_dist_with_bounds_neighbour(b, target):
    """The implementation does not assume that the inner boundary is inside the outer boundary."""
    o = expand_missing_outer_boundary(target[..., 4:].to(torch.float32))
    h = shrink_missing_inner_boundary_given_outer_boundary(
        target[..., :4].to(torch.float32), o
    )

    assert len(b.shape) == len(target.shape), (b.shape, target.shape)
    assert b.shape[-1] == 4, b.shape
    assert target.shape[-1] == 8, target.shape

    return torch.clamp(
        b[..., None, :],
        torch.cat([o[..., None, :, :2], h[..., None, :, 2:]], dim=-1),
        torch.cat([h[..., None, :, :2], o[..., None, :, 2:]], dim=-1),
    )


def _lp_loss_with_bounds_internal(o, h, b, *, p):
    # assert len(c.shape) == len(target.shape), (c.shape, target.shape)
    # assert pred.shape[-1] == 4, pred.shape
    # assert target.shape[-1] == 8, target.shape

    diffs = (
        torch.clamp(
            b,
            # When h is missing then h[..., 2:] = o[..., :2].
            torch.cat([o[..., :2], h[..., 2:]], dim=-1),
            # When h is missing then h[..., :2] = o[..., 2:].
            torch.cat([h[..., :2], o[..., 2:]], dim=-1),
        )
        - b
    )
    return LA.vector_norm(diffs, ord=p, dim=-1)


def _lp_loss_with_bounds(pred, target, *, p):
    """
    When p=1 like l1_loss with reduction="sum", but one comparison at a time.
    """
    a = expand_missing_outer_boundary(target[..., 4:].to(torch.float32))
    b = shrink_missing_inner_boundary_given_outer_boundary(
        target[..., :4].to(torch.float32), a
    )
    c = pred.to(torch.float32)
    return _lp_loss_with_bounds_internal(a, b, c, p=p)


def lp_loss_with_bounds_cxcywh(pred, target, *, p):
    """
    When p=1 like l1_loss with reduction="sum", but one comparison at a time.
    """
    o = expand_missing_outer_boundary_cxcywh(target[..., 4:].to(torch.float32))
    h = shrink_missing_inner_boundary_given_outer_boundary_cxcywh(
        target[..., :4].to(torch.float32), o
    )
    b = pred.to(torch.float32)

    # When h missing: h[..., 2:] = -o[..., 2:].
    s_x = torch.clamp(b[..., 2:], h[..., 2:], o[..., 2:])
    # When h missing: o[..., :2] - o[..., 2:] / 2 + s_x / 2
    l_c = torch.max(
        o[..., :2] - o[..., 2:] / 2 + s_x / 2, h[..., :2] + h[..., 2:] / 2 - s_x / 2
    )
    # When h missing: o[..., :2] + o[..., 2:] / 2 - s_x / 2
    u_c = torch.min(
        h[..., :2] - h[..., 2:] / 2 + s_x / 2, o[..., :2] + o[..., 2:] / 2 - s_x / 2
    )
    c_x = torch.clamp(b[..., :2], l_c, u_c)
    diffs = torch.cat([c_x - b[..., :2], s_x - b[..., 2:]], dim=-1)
    return LA.vector_norm(diffs, ord=p, dim=-1)


def cross_lp_loss_with_bounds_cxcywh_neighbour(b, target):
    """
    Amongst all the rectangles matching target, the small distance from pred to one of those
    rectangles.

    When p=1 like l1_loss with reduction="sum", but one comparison at a time.
    """
    o = expand_missing_outer_boundary_cxcywh(target[..., 4:].to(torch.float32))
    h = shrink_missing_inner_boundary_given_outer_boundary_cxcywh(
        target[..., :4].to(torch.float32), o
    )

    s_x = torch.clamp(b[..., None, 2:], h[..., None, :, 2:], o[..., None, :, 2:])
    l_c = torch.max(
        o[..., None, :, :2] - o[..., None, :, 2:] / 2 + s_x / 2,
        h[..., None, :, :2] + h[..., None, :, 2:] / 2 - s_x / 2,
    )
    u_c = torch.min(
        h[..., None, :, :2] - h[..., None, :, 2:] / 2 + s_x / 2,
        o[..., None, :, :2] + o[..., None, :, 2:] / 2 - s_x / 2,
    )
    c_x = torch.clamp(b[..., None, :2], l_c, u_c)
    # print("cross_lp_loss_with_bounds_cxcywh_neighbour: {}".format(pprint.pformat(locals())))
    return torch.cat([c_x, s_x], dim=-1)


def cross_lp_loss_with_bounds_cxcywh(pred, target, *, p):
    b = pred.to(torch.float32)
    x = cross_lp_loss_with_bounds_cxcywh_neighbour(b, target)
    return LA.vector_norm(x - b[..., None, :], ord=p, dim=-1)


# Target has two boxes for each entry, inner (b) and outer (a).
# When a has x1 > y1 or x2 > y2, then it is treated as missing.
# When b has x1 > y1 or x2 > y2, then it is treated as missing.
def non_differentiable_cross_bounded_box_iou_with_bounds(pred, target):
    a = target[..., 4:].to(torch.float32)
    b = target[..., :4].to(torch.float32)
    c = pred

    a = expand_missing_outer_boundary(a)
    b = shrink_missing_inner_boundary_given_outer_boundary(
        b,
        torch.tensor([-np.inf, -np.inf, np.inf, np.inf]),
    )

    assert len(c.shape) == len(target.shape), (c.shape, target.shape)
    assert pred.shape[-1] == 4, pred.shape
    assert target.shape[-1] == 8, target.shape

    assert (c[..., 2:] >= c[..., :2]).all()
    c_area = multidimensional_box_area(c)

    a_over_c_or_a = compute_a_over_c_or_a(a, c, c_area)
    c_and_b_over_b, c_or_b_area_over_eba = cross_factors(b, c, c_area)

    return a_over_c_or_a * c_and_b_over_b + c_or_b_area_over_eba - 1


def cross_factors(b, c, c_area):
    b_area, c_and_b_area, c_or_b_area = _cross_box_inter_union_with_known_area(
        (c, c_area), b
    )

    # Note that b invalid, i.e. (inf, inf, -inf, -inf) always results in c_contains_b.
    c_contains_b = (c[..., None, :2] <= b[..., None, :, :2]).all(dim=-1) & (
        b[..., None, :, 2:] <= c[..., None, 2:]
    ).all(dim=-1)

    # Missing b gets one.
    c_and_b_over_b = torch.where(
        c_contains_b,
        1,
        c_and_b_area / torch.where(b_area.bool(), b_area, 1)[..., None, :],
    )

    eba = _cross_enclosing_box_area(c, b)
    # TODO: Define c_or_b_area / eba as zero whenever eba is zero so that only c_contains_b can achieve the maximum value.
    c_or_b_area_over_eba = torch.where(c_contains_b, 1, c_or_b_area / eba)

    return c_and_b_over_b, c_or_b_area_over_eba


def compute_a_over_c_or_a(a, c, c_area):
    a_area, _, c_or_a_area = _cross_box_inter_union_with_known_area((c, c_area), a)
    a_contains_c = (a[..., None, :, :2] <= c[..., None, :2]).all(dim=-1) & (
        c[..., None, 2:] <= a[..., None, :, 2:]
    ).all(dim=-1)
    # a being the universe i.e. (-inf, -inf, inf, inf) implies a_contains_c.
    return torch.where(
        a_contains_c,
        1,
        torch.where(c_or_a_area.bool(), a_area[..., None, :] / c_or_a_area, 0),
    )


def cross_bounded_box_iou_with_bounds(pred, target):
    a = target[..., 4:].to(torch.float32)
    b = target[..., :4].to(torch.float32)
    c = pred

    a_is_present = is_present(a)
    a_is_present_expanded = expand_pred(a_is_present, c.shape)
    b_is_present = is_present(b)
    b_is_present_expanded = expand_pred(b_is_present, c.shape)

    a = a.to(torch.float32)
    b = b.to(torch.float32)
    c = pred

    assert len(c.shape) == len(target.shape), (c.shape, target.shape)
    assert pred.shape[-1] == 4, pred.shape
    assert target.shape[-1] == 8, target.shape

    assert (c[..., 2:] >= c[..., :2]).all()
    c_area = multidimensional_box_area(c)

    a_over_c_or_a = compute_a_over_c_or_a(a, c, c_area)
    c_and_b_over_b, c_or_b_area_over_eba = cross_factors(b, c, c_area)

    return (
        torch.where(a_is_present_expanded, a_over_c_or_a, 1)
        * torch.where(b_is_present_expanded, c_and_b_over_b, 1)
        + torch.where(b_is_present_expanded, c_or_b_area_over_eba, 1)
        - 1
    )


# def keep_two_drop_two(d, shape):
#     assert shape[-1] == d * 4
#     return torch.arange(0, d * 4)[torch.tensor([True, True, False, False]).repeat(d)].view(*([1] * (d - 1) + [d * 2])).expand(*(shape[:-1] + (d * 2, )))


def is_present_linear(boundary, *, alpha, beta, theta, phi):
    d = boundary.shape[-1] // 2
    assert boundary.shape[-1] == 2 * d
    phi_alpha_minus_theta_beta = phi * alpha - theta * beta
    l = phi * boundary[..., :d] - beta * boundary[..., d:]
    u = -theta * boundary[..., :d] + alpha * boundary[..., d:]
    # print("is_present_linear: {}".format(pprint.pformat(locals())))
    return (
        l <= u
        if phi_alpha_minus_theta_beta > 0
        else u <= l
        if phi_alpha_minus_theta_beta < 0
        else torch.ones_like(boundary[..., :d], dtype=torch.bool)
    )


def expand_missing_outer_boundary_linear(o, *, alpha, beta, theta, phi):
    d = o.shape[-1] // 2
    assert o.shape[-1] == 2 * d
    any_finite_value = 0
    first = torch.full_like(
        o[..., :d],
        # (alpha * (-np.inf) if alpha else 0) + (beta * np.inf if beta else 0)
        # if np.sign(alpha) * np.sign(beta) != 1
        # else any_finite_value,
        any_finite_value if alpha == beta else (-alpha + beta) * np.inf
    )
    second = torch.full_like(
        o[..., d:],
        # (theta * (-np.inf) if theta else 0) + (phi * np.inf if phi else 0)
        # if np.sign(theta) * np.sign(phi) != 1
        # else any_finite_value,
        any_finite_value if theta == phi else (-theta + phi) * np.inf
    )
    presence_indicator = is_present_linear(
        o, alpha=alpha, beta=beta, theta=theta, phi=phi
    )
    # print("expand_missing_outer_boundary_linear: {}".format(pprint.pformat(locals())))
    left = o[..., :d].where(presence_indicator, first)
    right = o[..., d:].where(presence_indicator, second)
    result = torch.cat((left, right), dim=-1)
    assert is_present_linear(result, alpha=alpha, beta=beta, theta=theta, phi=phi).all()
    return result


def shrink_missing_inner_boundary_given_outer_boundary_linear(
    h, o, *, alpha, beta, theta, phi
):
    assert h.shape == o.shape
    d = h.shape[-1] // 2
    assert h.shape[-1] == 2 * d

    presence_indicator = is_present_linear(
        h, alpha=alpha, beta=beta, theta=theta, phi=phi
    )

    alpha_theta_minus_beta_phi = alpha * theta - beta * phi

    alpha2_minus_beta2 = alpha**2 - beta**2
    first = (
        torch.zeros_like(o[..., d:])
        if alpha2_minus_beta2 == 0
        else alpha2_minus_beta2 * o[..., d:]
    ) - alpha_theta_minus_beta_phi * o[..., :d]

    theta2_minus_phi2 = theta**2 - phi**2
    second = alpha_theta_minus_beta_phi * o[..., d:] - (
        torch.zeros_like(o[..., :d])
        if theta2_minus_phi2 == 0
        else theta2_minus_phi2 * o[..., :d]
    )

    phi_alpha_minus_theta_beta = phi * alpha - theta * beta
    left = h[..., :d].where(presence_indicator, first / phi_alpha_minus_theta_beta)
    right = h[..., d:].where(presence_indicator, second / phi_alpha_minus_theta_beta)

    result = torch.cat((left, right), dim=-1)
    # print("shrink_hole_border_linear: {}".format(pprint.pformat(locals())))
    return result


def cross_l1_loss_linear_neighbour_unscaled(b, target, *, alpha, beta, theta, phi):
    assert abs(theta) >= abs(alpha), (theta, alpha)
    assert abs(phi) >= abs(beta), (phi, beta)

    # Bounds format:
    # 0: (alpha * l + beta * u)
    # 1: (theta * l + phi * u)

    # target = target.to(torch.float32)

    assert len(b.shape) == len(target.shape)
    assert target.shape[-1] == 2 * b.shape[-1]
    d = b.shape[-1] // 2
    assert b.shape[-1] == 2 * d

    h = target[..., : 2 * d].to(torch.float32)
    o = target[..., 2 * d :].to(torch.float32)

    # print("Before o: {}".format(o))
    o = expand_missing_outer_boundary_linear(
        o, alpha=alpha, beta=beta, theta=theta, phi=phi
    )
    # print("After o: {}".format(o))
    # print("Before h: {}".format(h))
    h = shrink_missing_inner_boundary_given_outer_boundary_linear(
        h, o, alpha=alpha, beta=beta, theta=theta, phi=phi
    )
    # print("After h: {}".format(h))
    target = torch.cat((h, o), dim=-1)

    # Clamp phi_alpha_minus_theta_beta * b[1]
    # Index 0 is h, index 1 is o.
    # 0::2 -> keep d, drop d
    # 1::2 -> drop d, keep d

    keep_d_drop_d_mask = torch.tensor(([True] * d + [False] * d) * 2)
    drop_d_keep_d_mask = ~keep_d_drop_d_mask
    # target.masked_select(keep_d_drop_d_mask).reshape(*(target.shape[:-1] + (d * 2, )))

    t02 = target[..., keep_d_drop_d_mask]  # (l^h, l^o)
    t12 = target[..., drop_d_keep_d_mask]  # (u^h, u^o)
    # theta * delta * (l^h_original, l^o_original).
    theta_delta_l = theta * (-beta * t12 + phi * t02)
    # phi * delta * (u^h_original, u^o_original).
    phi_delta_u = phi * (alpha * t12 - theta * t02)

    delta = phi * alpha - theta * beta

    sign_theta_delta = np.sign(theta) * np.sign(delta)
    sign_phi_delta = np.sign(phi) * np.sign(delta)

    lt = (
        theta_delta_l[
            ..., None, :, slice(d, 2 * d) if sign_theta_delta >= 0 else slice(0, d)
        ]
        + phi_delta_u[
            ..., None, :, slice(0, d) if sign_phi_delta >= 0 else slice(d, 2 * d)
        ]
    )
    ut = (
        theta_delta_l[
            ..., None, :, slice(0, d) if sign_theta_delta >= 0 else slice(d, 2 * d)
        ]
        + phi_delta_u[
            ..., None, :, slice(d, 2 * d) if sign_phi_delta >= 0 else slice(0, d)
        ]
    )

    assert (lt <= ut).all(), (lt, ut)
    t = torch.clamp(
        delta * b[..., :, None, d:],
        lt,
        ut,
    )

    # Clamp theta * phi * phi_alpha_minus_theta_beta * (alpha * l + beta * u).
    l_h_constraint = t * theta * beta + theta * delta * (
        phi * h[..., None, :, :d] - beta * h[..., None, :, d:]
    )
    l_o_constraint = t * theta * beta + theta * delta * (
        phi * o[..., None, :, :d] - beta * o[..., None, :, d:]
    )
    u_h_constraint = t * phi * alpha - phi * delta * (
        -theta * h[..., None, :, :d] + alpha * h[..., None, :, d:]
    )
    u_o_constraint = t * phi * alpha - phi * delta * (
        -theta * o[..., None, :, :d] + alpha * o[..., None, :, d:]
    )

    l = (l_o_constraint if theta >= 0 else l_h_constraint).max(
        u_o_constraint if phi >= 0 else u_h_constraint
    )
    u = (l_h_constraint if theta >= 0 else l_o_constraint).min(
        u_h_constraint if phi >= 0 else u_o_constraint
    )
    factor = theta * phi * delta
    # print("linear_neighbour_unscaled: {}".format(pprint.pformat(locals())))
    assert (l <= u).all(), (l, u)
    s = torch.clamp(factor * b[..., :, None, :d], l, u)
    return delta, t, factor, s


def cross_l1_loss_linear_neighbour(b, target, *, alpha, beta, theta, phi):
    (
        phi_alpha_minus_theta_beta,
        t,
        factor,
        s,
    ) = cross_l1_loss_linear_neighbour_unscaled(
        b, target, alpha=alpha, beta=beta, theta=theta, phi=phi
    )
    # print("cross_l1_loss_linear_neighbour: {}".format(pprint.pformat(locals())))
    return torch.cat([s / factor, t / phi_alpha_minus_theta_beta], dim=-1)


def cross_l1_loss_with_bounds_linear(pred, target, *, alpha, beta, theta, phi):
    b = pred.to(torch.float32)
    x = cross_l1_loss_linear_neighbour(
        b, target, alpha=alpha, beta=beta, theta=theta, phi=phi
    )
    diff = x - b[..., None, :]
    # print("cross_l1_loss_with_bounds_linear: {}".format(pprint.pformat(locals())))
    return LA.vector_norm(diff, ord=1, dim=-1)
