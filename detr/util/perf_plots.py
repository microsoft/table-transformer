import math
import argparse
import pathlib
import re
import collections
import pprint

import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser("Performance plots", add_help=False)
    parser.add_argument("--input_logs", type=pathlib.Path, nargs="*")
    parser.add_argument("--file_regex", type=str)
    parser.add_argument("--output_dir", type=pathlib.Path)
    return parser


def draw_plots():
    args = parser.parse_args()
    print(args)
    filename_pattern = re.compile(args.file_regex)
    print("filename_pattern: {}".format(filename_pattern))
    metrics_pattern = re.compile(
        "pubmed: AP50: (?P<ap50>[0-9.]+), AP75: (?P<ap75>[0-9.]+), AP: (?P<ap>[0-9.]+), AR: (?P<ar>[0-9.]+)"
    )
    print("metrics_pattern: {}".format(metrics_pattern))
    train_len_pattern = re.compile(r"train_len: (?P<tl>\d+)")

    d = collections.defaultdict(list)
    for filepath in args.input_logs:
        print(filepath)
        filename_match = filename_pattern.match(filepath.name)
        has_fraction_group = "fraction" in filename_pattern.groupindex
        has_all_object_segmented_group = (
            "all_objects_segmented" in filename_pattern.groupindex
        )
        has_kept_group = "kept" in filename_pattern.groupindex
        if has_fraction_group:
            fraction = "{:.2%}".format(float(filename_match.group("fraction")))
        epoch = int(filename_match.group("epoch"))
        assert epoch >= 1
        values = None
        with open(filepath, mode="rt") as f:
            for line in f:
                # If "fraction" group is missing then read train_len from "train_len: 50000 batch_size: 2 args.fused: None".
                train_len_match = train_len_pattern.match(line)
                if train_len_match and not has_fraction_group:
                    fraction = "images: {}".format(train_len_match.group("tl"))
                metrics_match = metrics_pattern.match(line)
                if metrics_match:
                    values = {
                        key: float(value)
                        for key, value in metrics_match.groupdict().items()
                    }
        if values:
            v = d[
                "{}{}{}".format(
                    "boxes: {}{}; ".format(
                        filename_match.group("kept").rjust(2),
                        ""
                        if not has_all_object_segmented_group
                        else " (equiv)"
                        if filename_match.group("all_objects_segmented")
                        else " per image",
                    )
                    if has_kept_group and filename_match.group("kept")
                    else "",
                    fraction,
                    ""
                    if not has_all_object_segmented_group
                    else " (all boxes)"
                    if filename_match.group("all_objects_segmented")
                    else ""
                    if has_kept_group and filename_match.group("kept")
                    else " (random box subset)",
                )
            ]
            v.extend([None] * (epoch - len(v)))
            v[epoch - 1] = (filepath, values)
    print(pprint.pformat(d))

    epochs = max((len(v) for v in d.values()))
    # markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    markers = [
        "1",
        ".",
        "+",
        "x",
        "*",
        "o",
        "v",
        "s",
        "^",
        "<",
        ">",
        "8",
        "p",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
    ]
    for key in ("ap50", "ap75", "ap", "ar"):
        fig = plt.figure()
        ax = fig.subplots()  # add_subplot(111, aspect="equal")
        ax.set_xlim(0.75, epochs + 0.25)
        ax.set_xticks(list(range(1, epochs + 1)))
        # ax.set_ylim(0, 1)
        s = math.ceil(math.sqrt(len(d)))
        for index, (fraction, v) in enumerate(d.items()):
            print(fraction)
            x = [i + 1 for i in range(epochs) if i < len(v) and v[i] is not None]
            y = [value[key] for _, value in v if value is not None]
            ax.plot(
                x,
                y,
                label=fraction,
                linestyle=(0, (1 + index // s, 1 + index % s)),
                # linestyle='None',
                # markersize=5,
                marker=markers[index],
            )
            ax.set(xlabel="Epoch", ylabel=key.upper())
        ax.legend(loc="lower right")
        # ax.set_title(key.upper())
        # fig.legend()
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            image_path = pathlib.Path(args.output_dir, "{}_fractions.svg".format(key))
            print(image_path)
            fig.savefig(
                image_path,
                # bbox_extra_artists=(lgd,),
                bbox_inches="tight",
            )
            plt.close()


if __name__ == "__main__":
    parser = get_args_parser()
    draw_plots()
