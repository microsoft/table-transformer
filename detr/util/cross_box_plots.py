import torch
from torch import linalg as LA
import box_ops
import math
import numpy
import random

import matplotlib.pyplot as plt
from matplotlib import patches


def generate(a, upsample, n):
    # 16: 233323
    # 12: 0, 8(!)
    torch.manual_seed(11)
    numpy.random.seed(879723139)
    random.seed(3023867716)
    x_low, x_high = (a[0] * 2).floor(), (a[2] * 2).ceil()
    y_low, y_high = (a[1] * 2).floor(), (a[3] * 2).ceil()
    c = torch.empty((n * upsample + 1, 4), dtype=int)
    c[:, 0::2] = torch.randint(low=x_low, high=x_high + 1, size=(n * upsample + 1, 2))
    c[:, 1::2] = torch.randint(low=y_low, high=y_high + 1, size=(n * upsample + 1, 2))
    c[:, 0], c[:, 2] = torch.min(c[:, 0], c[:, 2]), torch.max(c[:, 0], c[:, 2])
    c[:, 1], c[:, 3] = torch.min(c[:, 1], c[:, 3]), torch.max(c[:, 1], c[:, 3])
    return torch.tensor([x_low, y_low, x_high, y_high]), c


def plot_rectangles(boxes, values, a, b, frame, filename):
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, aspect="equal")
    x_low, y_low, x_high, y_high = frame.unbind()
    lw_offset, lw_factor = 2, 2
    padding = math.ceil((lw_offset + 1) / lw_factor / 2)
    xlim = torch.tensor([x_low - padding - 1, x_high + padding + 1])
    ax.set_xlim(xlim)
    ylim = torch.tensor([y_low - padding - 1, y_high + padding + 1])
    ax.set_ylim(ylim)

    boundary_boxes = [
        b if box_ops.is_present(b) else None,
        a if box_ops.is_present(a) else None,
        torch.tensor([xlim[0], ylim[0], a[0], ylim[1]])
        if box_ops.is_present(a)
        else None,
        torch.tensor([xlim[0], ylim[0], xlim[1], a[1]])
        if box_ops.is_present(a)
        else None,
        torch.tensor([xlim[0], a[3], xlim[1], ylim[1]])
        if box_ops.is_present(a)
        else None,
        torch.tensor([a[2], ylim[0], xlim[1], ylim[1]])
        if box_ops.is_present(a)
        else None,
    ]

    for i, box in enumerate(boundary_boxes):
        if box is not None:
            ax.add_patch(
                patches.Rectangle(
                    box[:2].cpu(),
                    (box[2] - box[0]).item(),
                    (box[3] - box[1]).item(),
                    edgecolor="dimgray",
                    facecolor="white",
                    fill=True,  # remove background
                    zorder=-2 - min(i, 2),
                    linewidth=i < 2,
                    hatch="/" if not i else None if i == 1 else "\\",
                )
            )

    for i, (box, value) in enumerate(zip(boxes, values)):
        if (box[:2] == box[2:]).all():
            # Matplotlib can draw a rectangle with one 0 side, but not with both 0.
            ax.scatter(
                box[0].item(),
                box[1].item(),
                s=min_side(frame) / len(boxes) / 8,
                color=plt.cm.brg((1 + value.item()) / 2),
                alpha=0.8,
                marker="x",
                clip_on=False,
                linewidth=(lw_offset + value.item()) * lw_factor,
                linestyle=(0, [i + 1, 1]),
                zorder=value.item(),
                label="{:.2f}".format(value.item()),
            )
        else:
            ax.add_patch(
                patches.Rectangle(
                    box[:2].cpu(),
                    (box[2] - box[0]).item(),
                    (box[3] - box[1]).item(),
                    edgecolor=plt.cm.brg((1 + value.item()) / 2),
                    alpha=0.8,
                    fill=False,
                    linewidth=(lw_offset + value.item()) * lw_factor,
                    linestyle=(0, [i + 1, 1]),
                    zorder=value.item(),
                    label="{:.2f}".format(value.item()),
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    box[:2].cpu(),
                    (box[2] - box[0]).item(),
                    (box[3] - box[1]).item(),
                    edgecolor=plt.cm.brg((1 + value.item()) / 2),
                    alpha=0.8,
                    fill=False,
                    linewidth=1,
                    zorder=value.item(),
                )
            )

    handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # lgd = ax.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(1.05, +1.05))
    lgd = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(1.05, +1.05))
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticks([])
    ax.set_yticks([])
    print(filename)
    fig.savefig(
        filename,
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    plt.close()


def center_of_box(box):
    return torch.stack([box[:2], box[2:]]).mean(dim=0, dtype=float)


def min_side(box):
    return (box[2:] - box[:2]).min()


def concentric_boxes(frame, n):
    center = center_of_box(frame)
    lengths = torch.linspace(min_side(frame) - 2, 0, n)
    expanded_center = torch.cat([center, center])[None, :].expand(n, 4)
    factors = torch.tensor([-1, -1, 1, 1])[None, :].expand(n, 4)
    lengths_expanded = lengths[:, None].expand(n, 4)
    return expanded_center + factors * lengths_expanded / 2


def plot_boxes(
    function_name,
    selected_boxes_and_random_box_values,
    cb_and_concentric_box_values,
    a,
    b,
    frame,
    label,
):
    selected_boxes, (random_box_values, random_box_neighbours) = selected_boxes_and_random_box_values
    plot_rectangles(
        selected_boxes,
        random_box_values,
        a,
        b,
        frame,
        "/tmp/plots/random_{}_{}.svg".format(label, function_name),
    )
    if random_box_neighbours is not None:
        plot_rectangles(
            random_box_neighbours.squeeze(dim=-2),
            random_box_values,
            a,
            b,
            frame,
            "/tmp/plots/random_{}_{}_neighbour.svg".format(label, function_name)
        )
    cb, (concentric_box_values, concentric_box_neighbours) = cb_and_concentric_box_values
    plot_rectangles(
        cb,
        concentric_box_values,
        a,
        b,
        frame,
        "/tmp/plots/concentric_{}_{}.svg".format(label, function_name),
    )
    if concentric_box_neighbours is not None:
        plot_rectangles(
            concentric_box_neighbours.squeeze(dim=-2),
            concentric_box_values,
            a,
            b,
            frame,
            "/tmp/plots/concentric_{}_{}_neighbour.svg".format(label, function_name)
        )


def draw_plots_for_function(f, function_name, selected_boxes, cb, a, b, frame):
    # print("selected_boxes: {} cb: {} a: {} b: {}".format(selected_boxes, cb, a, b))
    random_box_values = f(selected_boxes, torch.cat((b, a))[None, :])
    # print("random_box_values: {}".format(random_box_values))
    concentric_box_values = f(cb, torch.cat((b, a))[None, :])
    # print("concentric_box_values: {}".format(concentric_box_values))
    plot_boxes(
        function_name,
        (selected_boxes, random_box_values),
        (cb, concentric_box_values),
        a,
        b,
        frame,
        "rectangles",
    )

    missing_hole_box = torch.tensor([1, -1, 0, 1])
    random_box_values = f(selected_boxes, torch.cat((missing_hole_box, a))[None, :])
    concentric_box_values = f(cb, torch.cat((missing_hole_box, a))[None, :])
    plot_boxes(
        function_name,
        (selected_boxes, random_box_values),
        (cb, concentric_box_values),
        a,
        missing_hole_box,
        frame,
        "outer",
    )

    missing_outer_box = torch.tensor([1, -1, 0, 1])
    random_box_values = f(selected_boxes, torch.cat((b, missing_outer_box))[None, :])
    concentric_box_values = f(cb, torch.cat((b, missing_outer_box))[None, :])
    plot_boxes(
        function_name,
        (selected_boxes, random_box_values),
        (cb, concentric_box_values),
        missing_outer_box,
        b,
        frame,
        "hole",
    )

    hole_box = torch.cat([center_of_box(a), center_of_box(a)])
    random_box_values = f(selected_boxes, torch.cat((hole_box, a))[None, :])
    concentric_box_values = f(cb, torch.cat((hole_box, a))[None, :])
    plot_boxes(
        function_name,
        (selected_boxes, random_box_values),
        (cb, concentric_box_values),
        a,
        hole_box,
        frame,
        "empty_hole_in_center_of_outer",
    )

    missing_outer_box = torch.tensor([1, -1, 0, 1])
    missing_hole_box = torch.tensor([1, -1, 0, 1])
    random_box_values = f(
        selected_boxes, torch.cat((missing_hole_box, missing_outer_box))[None, :]
    )
    concentric_box_values = f(
        cb, torch.cat((missing_hole_box, missing_outer_box))[None, :]
    )
    plot_boxes(
        function_name,
        (selected_boxes, random_box_values),
        (cb, concentric_box_values),
        missing_outer_box,
        missing_hole_box,
        frame,
        "unconstrained",
    )

    empty_outer_box = torch.cat([center_of_box(a), center_of_box(a)])
    missing_hole_box = torch.tensor([1, -1, 0, 1])
    random_box_values = f(
        selected_boxes, torch.cat((missing_hole_box, empty_outer_box))[None, :]
    )
    concentric_box_values = f(
        cb, torch.cat((missing_hole_box, empty_outer_box))[None, :]
    )
    plot_boxes(
        function_name,
        (selected_boxes, random_box_values),
        (cb, concentric_box_values),
        missing_outer_box,
        missing_hole_box,
        frame,
        "empty_outer",
    )

    missing_outer_box = torch.tensor([1, -1, 0, 1])
    empty_hole_box = torch.cat([center_of_box(b), center_of_box(b)])
    random_box_values = f(
        selected_boxes, torch.cat((empty_hole_box, missing_outer_box))[None, :]
    )
    concentric_box_values = f(
        cb, torch.cat((empty_hole_box, missing_outer_box))[None, :]
    )
    plot_boxes(
        function_name,
        (selected_boxes, random_box_values),
        (cb, concentric_box_values),
        missing_outer_box,
        empty_hole_box,
        frame,
        "empty_hole",
    )

    empty_outer_box = torch.cat([center_of_box(a), center_of_box(a)])
    empty_hole_box = torch.cat([center_of_box(b), center_of_box(b)])
    random_box_values = f(
        selected_boxes, torch.cat((empty_hole_box, empty_outer_box))[None, :]
    )
    concentric_box_values = f(cb, torch.cat((empty_hole_box, empty_outer_box))[None, :])
    plot_boxes(
        function_name,
        (selected_boxes, random_box_values),
        (cb, concentric_box_values),
        empty_outer_box,
        empty_hole_box,
        frame,
        "empty_hole_in_center_of_empty_outer",
    )


# def normalize_minus_one_to_one(a):
#     a -= a.min(dim=0, keepdim=True)[0]
#     a /= a.max(dim=0, keepdim=True)[0]
#     return torch.nan_to_num(a * 2 - 1)


def normalize_nonnegative_to_minus_one_to_one(a):
    return torch.nan_to_num(1 - 2 * a / a.max())

def _cross_bounded_box_iou_with_bounds_distance_neighbour(pred, target):
    distances = box_ops.cross_bounded_box_iou_with_bounds(
        pred, target)
    return distances, None


def _cross_dist_with_bounds_distance_neighbour(pred, target):
    b = pred.to(torch.float32)
    p = 1
    x = box_ops.cross_dist_with_bounds_neighbour(
        pred, target)
    distances = LA.vector_norm(x - b[..., None, :], ord=p, dim=-1)
    return normalize_nonnegative_to_minus_one_to_one(distances), x


def _cross_lp_loss_with_bounds_cxcywh_distance_neighbour(pred, target):
    b = box_ops.box_xyxy_to_cxcywh(pred)
    p = 1
    x = box_ops.cross_lp_loss_with_bounds_cxcywh_neighbour(
        b,
        box_ops.box_xyxy_to_cxcywh_with_bounds(target),
    )
    distances = LA.vector_norm(x - b[..., None, :], ord=p, dim=-1)
    return (
        normalize_nonnegative_to_minus_one_to_one(distances),
        box_ops.box_cxcywh_to_xyxy(x)
        )


def draw_plots():
    factor = 1000
    a = torch.tensor([-3, -3, 3, 3]) * factor
    b = torch.tensor([-1, -1, 1, 1]) * factor
    stretch_x, stretch_y = 3, 2
    a[0::2] *= stretch_x
    b[0::2] *= stretch_x
    a[1::2] *= stretch_y
    b[1::2] *= stretch_y
    n = 4
    upsample = 5000
    frame, c = generate(a, upsample, n)
    cross_values = box_ops.cross_bounded_box_iou_with_bounds(
        c, torch.cat((b, a))[None, :]
    )

    sorted_indices = torch.argsort(cross_values, dim=0, stable=True)
    indices = sorted_indices[::upsample]
    selected_boxes = torch.take_along_dim(c, indices, 0)
    cb = concentric_boxes(frame, 8)

    draw_plots_for_function(
        _cross_dist_with_bounds_distance_neighbour,
        "cross_dist",
        selected_boxes,
        cb,
        a,
        b,
        frame,
    )
    draw_plots_for_function(
        _cross_lp_loss_with_bounds_cxcywh_distance_neighbour,
        "cross_dist_cxcywh",
        selected_boxes,
        cb,
        a,
        b,
        frame,
    )
    draw_plots_for_function(
        _cross_bounded_box_iou_with_bounds_distance_neighbour,
        "cross_riou",
        selected_boxes,
        cb,
        a,
        b,
        frame,
    )


if __name__ == "__main__":
    draw_plots()
