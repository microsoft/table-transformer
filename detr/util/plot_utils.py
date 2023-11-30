"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
import os
import errno

from pathlib import Path, PurePath

sys.path.append("src")

# import table_datasets

def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [PurePath(f.name).parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [PurePath(f.name).stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color_all, color_large, name in zip(files,
                                               sns.color_palette("Blues", n_colors=len(files)),
                                               sns.color_palette("Reds", n_colors=len(files)),
                                               names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        # print("precision: {}".format(precision.shape))
        recall = data['params'].recThrs
        # print('Recall: {}', recall)
        assert precision.shape[1] == len(recall)
        scores = data['scores']
        assert precision.shape == scores.shape
        overlap_half = 0
        object_area_ranges_all = 0
        object_area_ranges_large = 3
        category_id =  0 # table, without table rotated
        # print(precision[:, :, :, object_area_ranges_all, :])
        # q = torch.tensor([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0], dtype=float)
        # print("Quantiles: {}".format(torch.quantile(torch.tensor(precision[0, :, :2, 0, -1]), q)))
        # take precision for all classes, all areas and 100 detections
        precision_all = precision[overlap_half, :, category_id, object_area_ranges_all, -1]
        precision_large = precision[overlap_half, :, category_id, object_area_ranges_large, -1]
        assert precision_all.shape == (len(recall), )
        # .mean(1)
        for r in (0, 0.01, 0.05, 0.5, 0.9, 0.95, 0.99, 1):
            print('precision_all(recall={}): {}'.format(r, precision_all[abs(recall - r) < 1e-6]))
        for r in (0, 0.01, 0.05, 0.5, 0.9, 0.95, 0.99, 1):
            print('precision_large(recall={}): {}'.format(r, precision_large[abs(recall - r) < 1e-6]))
        scores_all = scores[overlap_half, :, category_id, object_area_ranges_all, -1]
        scores_large = scores[overlap_half, :, category_id, object_area_ranges_large, -1]
        prec = precision_all.mean()
        rec = data['recall'][overlap_half, category_id, object_area_ranges_all, -1]
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores_all.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        # Only non-rotated tables.
        axs[0].plot(recall, precision_all, c=color_all)
        axs[0].plot(recall, precision_large, c=color_large)
        axs[0].set_xlabel('Recall')
        axs[0].set_ylabel('Precision')
        axs[1].plot(recall, scores_all, c=color_all)
        axs[1].plot(recall, scores_large, c=color_large)
        axs[1].set_xlabel('Recall')
        axs[1].set_ylabel('Score')

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_eval_path', type=argparse.FileType('rb'), nargs='+')
    parser.add_argument(
        "--plot_path",
        required=True,
        type=PurePath
    )
    return parser.parse_args()


def make_image(coco_eval_files, plot_path):
    try:
        os.makedirs(os.path.dirname(plot_path))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
    fig, axs = plot_precision_recall(coco_eval_files)
    fig.savefig(plot_path)

def main():
    args = get_args()
    make_image(args.coco_eval_path, args.plot_path)

if __name__ == '__main__':
    main()
