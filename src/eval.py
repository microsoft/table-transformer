"""
Copyright (C) 2021 Microsoft Corporation
"""
import os
import sys
from collections import Counter
import json
import statistics as stat
from datetime import datetime

import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fitz import Rect

sys.path.append("../detr")
from engine import evaluate
import postprocess
import grits
from grits import grits_con, grits_top, grits_loc


def objects_to_cells(bboxes, labels, scores, page_tokens, structure_class_names, structure_class_thresholds, structure_class_map):
    bboxes, scores, labels = postprocess.apply_class_thresholds(bboxes, labels, scores,
                                            structure_class_names,
                                            structure_class_thresholds)

    table_objects = []
    for bbox, score, label in zip(bboxes, scores, labels):
        table_objects.append({'bbox': bbox, 'score': score, 'label': label})
        
    table = {'objects': table_objects, 'page_num': 0} 
    
    table_class_objects = [obj for obj in table_objects if obj['label'] == structure_class_map['table']]
    if len(table_class_objects) > 1:
        table_class_objects = sorted(table_class_objects, key=lambda x: x['score'], reverse=True)
    try:
        table_bbox = list(table_class_objects[0]['bbox'])
    except:
        table_bbox = (0,0,1000,1000)
    
    tokens_in_table = [token for token in page_tokens if postprocess.iob(token['bbox'], table_bbox) >= 0.5]
    
    # Determine the table cell structure from the objects
    table_structures, cells, confidence_score = postprocess.objects_to_cells(table, table_objects, tokens_in_table,
                                                                    structure_class_names,
                                                                    structure_class_thresholds)
    
    return table_structures, cells, confidence_score


def cells_to_adjacency_pair_list(cells, key='cell_text'):
    # Index the cells by their grid coordinates
    cell_nums_by_coordinates = dict()
    for cell_num, cell in enumerate(cells):
        for row_num in cell['row_nums']:
            for column_num in cell['column_nums']:
                cell_nums_by_coordinates[(row_num, column_num)] = cell_num

    # Count the number of unique rows and columns
    row_nums = set()
    column_nums = set()
    for cell in cells:
        for row_num in cell['row_nums']:
            row_nums.add(row_num)
        for column_num in cell['column_nums']:
            column_nums.add(column_num)
    num_rows = len(row_nums)
    num_columns = len(column_nums)

    # For each cell, determine its next neighbors
    # - For every row the cell occupies, what is the first cell to the right with text that
    #   also occupies that row
    # - For every column the cell occupies, what is the first cell below with text that
    #   also occupies that column
    adjacency_list = []
    adjacency_bboxes = []
    for cell1_num, cell1 in enumerate(cells):
        # Skip blank cells
        if cell1['cell_text'] == '':
            continue

        adjacent_cell_props = {}
        max_column = max(cell1['column_nums'])
        max_row = max(cell1['row_nums'])

        # For every column the cell occupies...
        for column_num in cell1['column_nums']:
            # Start from the next row and stop when we encounter a non-blank cell
            # This cell is considered adjacent
            for current_row in range(max_row+1, num_rows):
                cell2_num = cell_nums_by_coordinates[(current_row, column_num)]
                cell2 = cells[cell2_num]
                if not cell2['cell_text'] == '':
                    adj_bbox = [(max(cell1['bbox'][0], cell2['bbox'][0])+min(cell1['bbox'][2], cell2['bbox'][2]))/2-3,
                                cell1['bbox'][3],
                                (max(cell1['bbox'][0], cell2['bbox'][0])+min(cell1['bbox'][2], cell2['bbox'][2]))/2+3,
                                cell2['bbox'][1]]
                    adjacent_cell_props[cell2_num] = ('V', current_row - max_row - 1,
                                                      adj_bbox)
                    break

        # For every row the cell occupies...
        for row_num in cell1['row_nums']:
            # Start from the next column and stop when we encounter a non-blank cell
            # This cell is considered adjacent
            for current_column in range(max_column+1, num_columns):
                cell2_num = cell_nums_by_coordinates[(row_num, current_column)]
                cell2 = cells[cell2_num]
                if not cell2['cell_text'] == '':
                    adj_bbox = [cell1['bbox'][2],
                                (max(cell1['bbox'][1], cell2['bbox'][1])+min(cell1['bbox'][3], cell2['bbox'][3]))/2-3,
                                cell2['bbox'][0],
                                (max(cell1['bbox'][1], cell2['bbox'][1])+min(cell1['bbox'][3], cell2['bbox'][3]))/2+3]
                    adjacent_cell_props[cell2_num] = ('H', current_column - max_column - 1,
                                                      adj_bbox)
                    break

        for adjacent_cell_num, props in adjacent_cell_props.items():
            cell2 = cells[adjacent_cell_num]
            adjacency_list.append((cell1['cell_text'], cell2['cell_text'], props[0], props[1]))
            adjacency_bboxes.append(props[2])

    return adjacency_list, adjacency_bboxes


def cells_to_adjacency_pair_list_with_blanks(cells, key='cell_text'):
    # Index the cells by their grid coordinates
    cell_nums_by_coordinates = dict()
    for cell_num, cell in enumerate(cells):
        for row_num in cell['row_nums']:
            for column_num in cell['column_nums']:
                cell_nums_by_coordinates[(row_num, column_num)] = cell_num

    # Count the number of unique rows and columns
    row_nums = set()
    column_nums = set()
    for cell in cells:
        for row_num in cell['row_nums']:
            row_nums.add(row_num)
        for column_num in cell['column_nums']:
            column_nums.add(column_num)
    num_rows = len(row_nums)
    num_columns = len(column_nums)

    # For each cell, determine its next neighbors
    # - For every row the cell occupies, what is the next cell to the right
    # - For every column the cell occupies, what is the next cell below
    adjacency_list = []
    adjacency_bboxes = []
    for cell1_num, cell1 in enumerate(cells):
        adjacent_cell_props = {}
        max_column = max(cell1['column_nums'])
        max_row = max(cell1['row_nums'])

        # For every column the cell occupies...
        for column_num in cell1['column_nums']:
            # The cell in the next row is adjacent
            current_row = max_row + 1
            if current_row >= num_rows:
                continue
            cell2_num = cell_nums_by_coordinates[(current_row, column_num)]
            cell2 = cells[cell2_num]
            adj_bbox = [(max(cell1['bbox'][0], cell2['bbox'][0])+min(cell1['bbox'][2], cell2['bbox'][2]))/2-3,
                        cell1['bbox'][3],
                        (max(cell1['bbox'][0], cell2['bbox'][0])+min(cell1['bbox'][2], cell2['bbox'][2]))/2+3,
                        cell2['bbox'][1]]
            adjacent_cell_props[cell2_num] = ('V', current_row - max_row - 1,
                                              adj_bbox)

        # For every row the cell occupies...
        for row_num in cell1['row_nums']:
            # The cell in the next column is adjacent
            current_column = max_column + 1
            if current_column >= num_columns:
                continue
            cell2_num = cell_nums_by_coordinates[(row_num, current_column)]
            cell2 = cells[cell2_num]
            adj_bbox = [cell1['bbox'][2],
                        (max(cell1['bbox'][1], cell2['bbox'][1])+min(cell1['bbox'][3], cell2['bbox'][3]))/2-3,
                        cell2['bbox'][0],
                        (max(cell1['bbox'][1], cell2['bbox'][1])+min(cell1['bbox'][3], cell2['bbox'][3]))/2+3]
            adjacent_cell_props[cell2_num] = ('H', current_column - max_column - 1,
                                              adj_bbox)

        for adjacent_cell_num, props in adjacent_cell_props.items():
            cell2 = cells[adjacent_cell_num]
            adjacency_list.append((cell1['cell_text'], cell2['cell_text'], props[0], props[1]))
            adjacency_bboxes.append(props[2])

    return adjacency_list, adjacency_bboxes


def dar_con(true_adjacencies, pred_adjacencies):
    """
    Directed adjacency relations (DAR) metric, which uses exact match
    between adjacent cell text content.
    """

    true_c = Counter()
    true_c.update([elem for elem in true_adjacencies])

    pred_c = Counter()
    pred_c.update([elem for elem in pred_adjacencies])

    num_true_positives = (sum(true_c.values()) - sum((true_c - pred_c).values()))

    fscore, precision, recall = grits.compute_fscore(num_true_positives,
                                               len(true_adjacencies),
                                               len(pred_adjacencies))

    return recall, precision, fscore


def dar_con_original(true_cells, pred_cells):
    """
    Original DAR metric, where blank cells are disregarded.
    """
    true_adjacencies, _ = cells_to_adjacency_pair_list(true_cells)
    pred_adjacencies, _ = cells_to_adjacency_pair_list(pred_cells)

    return dar_con(true_adjacencies, pred_adjacencies)


def dar_con_new(true_cells, pred_cells):
    """
    New version of DAR metric where blank cells count.
    """
    true_adjacencies, _ = cells_to_adjacency_pair_list_with_blanks(true_cells)
    pred_adjacencies, _ = cells_to_adjacency_pair_list_with_blanks(pred_cells)

    return dar_con(true_adjacencies, pred_adjacencies)


def compute_metrics(mode, true_bboxes, true_labels, true_scores, true_cells,
                    pred_bboxes, pred_labels, pred_scores, pred_cells):
    """
    Compute the collection of table structure recognition metrics given
    the ground truth and predictions as input.

    - bboxes, labels, and scores are required to compute GriTS_RawLoc, which
      is GriTS_Loc but on unprocessed bounding boxes, compared with the dilated
      ground truth bounding boxes the model is trained on.
    - Otherwise, only true_cells and pred_cells are needed.
    """
    metrics = {}

    # Compute grids/matrices for comparison
    true_relspan_grid = np.array(grits.cells_to_relspan_grid(true_cells))
    true_bbox_grid = np.array(grits.cells_to_grid(true_cells, key='bbox'))
    true_text_grid = np.array(grits.cells_to_grid(true_cells, key='cell_text'), dtype=object)
    pred_relspan_grid = np.array(grits.cells_to_relspan_grid(pred_cells))
    pred_bbox_grid = np.array(grits.cells_to_grid(pred_cells, key='bbox'))
    pred_text_grid = np.array(grits.cells_to_grid(pred_cells, key='cell_text'), dtype=object)

    # Compute GriTS_Top (topology)
    (metrics['grits_top'],
     metrics['grits_precision_top'],
     metrics['grits_recall_top'],
     metrics['grits_top_upper_bound']) = grits_top(true_relspan_grid,
                                                   pred_relspan_grid)

    # Compute GriTS_Loc (location)
    (metrics['grits_loc'],
     metrics['grits_precision_loc'],
     metrics['grits_recall_loc'],
     metrics['grits_loc_upper_bound']) = grits_loc(true_bbox_grid,
                                                   pred_bbox_grid)

    # Compute GriTS_Con (text content)
    (metrics['grits_con'],
     metrics['grits_precision_con'],
     metrics['grits_recall_con'],
     metrics['grits_con_upper_bound']) = grits_con(true_text_grid,
                                                   pred_text_grid)

    # Compute content accuracy
    metrics['acc_con'] = int(metrics['grits_con'] == 1)

    if mode == 'grits-all':
        # Compute grids/matrices for comparison
        true_cell_dilatedbbox_grid = np.array(grits.output_to_dilatedbbox_grid(true_bboxes, true_labels, true_scores))
        pred_cell_dilatedbbox_grid = np.array(grits.output_to_dilatedbbox_grid(pred_bboxes, pred_labels, pred_scores))

        # Compute GriTS_RawLoc (location using unprocessed bounding boxes)
        (metrics['grits_rawloc'],
        metrics['grits_precision_rawloc'],
        metrics['grits_recall_rawloc'],
        metrics['grits_rawloc_upper_bound']) = grits_loc(true_cell_dilatedbbox_grid,
                                                        pred_cell_dilatedbbox_grid)

        # Compute original DAR (directed adjacency relations) metric
        (metrics['dar_recall_con_original'], metrics['dar_precision_con_original'],
        metrics['dar_con_original']) = dar_con_original(true_cells, pred_cells)

        # Compute updated DAR (directed adjacency relations) metric
        (metrics['dar_recall_con'], metrics['dar_precision_con'],
        metrics['dar_con']) = dar_con_new(true_cells, pred_cells)

    return metrics


def compute_statistics(structures, cells):
    statistics = {}
    statistics['num_rows'] = len(structures['rows'])
    statistics['num_columns'] = len(structures['columns'])
    statistics['num_cells'] = len(cells)
    statistics['num_spanning_cells'] = len([cell for cell in cells if len(cell['row_nums']) > 1
                                            or len(cell['column_nums']) > 1])
    header_rows = set()
    for cell in cells:
        if cell['header']:
            header_rows = header_rows.union(set(cell['row_nums']))
    statistics['num_header_rows'] = len(header_rows)
    row_heights = [float(row['bbox'][3]-row['bbox'][1]) for row in structures['rows']]
    if len(row_heights) >= 2:
        statistics['row_height_coefficient_of_variation'] = stat.stdev(row_heights) / stat.mean(row_heights)
    else:
        statistics['row_height_coefficient_of_variation'] = 0
    column_widths = [float(column['bbox'][2]-column['bbox'][0]) for column in structures['columns']]
    if len(column_widths) >= 2:
        statistics['column_width_coefficient_of_variation'] = stat.stdev(column_widths) / stat.mean(column_widths)
    else:
        statistics['column_width_coefficient_of_variation'] = 0

    return statistics


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def get_bbox_decorations(label, score):
    colors = [
        'brown', 'red', 'blue', 'magenta', 'cyan', 'green', 'orange', 'green',
        'orange', 'yellow', 'brown', 'red', 'blue', 'magenta', 'cyan', 'green',
        'orange', 'green', 'orange', 'yellow'
    ]
    if label == 0 or label == 8:
        alpha = 0
        linewidth = 3
    elif label == 3:
        alpha = score / 3
        linewidth = 3
    elif label == 4 or label == 5:
        alpha = score / 3
        linewidth = 4
    else:
        alpha = score / 9
        linewidth = 2

    color = colors[label]

    return color, alpha, linewidth


def plot_graph(metric_1, metric_2, metric_1_name, metric_2_name):
    plt.scatter(metric_1, metric_2, s=40, c='red', marker='o')
    plt.title(metric_1_name + " vs. " + metric_2_name)
    plt.xlim([0.5, 1])
    plt.ylim([0.5, 1])
    plt.plot([0, 1], [0, 1])
    plt.xlabel(metric_1_name)
    plt.ylabel(metric_2_name)
    plt.gcf().set_size_inches((8, 8))
    plt.show()


def compute_metrics_summary(sample_metrics, mode):
    """
    Print a formatted summary of the table structure recognition metrics
    averaged over all samples.
    """

    metrics_summary = {}

    metric_names = ['acc_con', 'grits_top', 'grits_con', 'grits_loc']
    if mode == 'grits-all':
        metric_names += ['grits_rawloc', 'dar_con_original', 'dar_con']

    simple_samples = [entry for entry in sample_metrics if entry['num_spanning_cells'] == 0]
    metrics_summary['simple'] = {'num_tables': len(simple_samples)}
    if len(simple_samples) > 0:
        for metric_name in metric_names:
            metrics_summary['simple'][metric_name] = np.mean([elem[metric_name] for elem in simple_samples])

    complex_samples = [entry for entry in sample_metrics if entry['num_spanning_cells'] > 0]
    metrics_summary['complex'] = {'num_tables': len(complex_samples)}
    if len(complex_samples) > 0:
        for metric_name in metric_names:
            metrics_summary['complex'][metric_name] = np.mean([elem[metric_name] for elem in complex_samples])

    metrics_summary['all'] = {'num_tables': len(sample_metrics)}
    if len(sample_metrics) > 0:
        for metric_name in metric_names:
            metrics_summary['all'][metric_name] = np.mean([elem[metric_name] for elem in sample_metrics])

    return metrics_summary


def print_metrics_summary(metrics_summary):
    """
    Print a formatted summary of the table structure recognition metrics
    averaged over all samples.
    """

    print('-' * 100)
    for table_type in ['simple', 'complex', 'all']:
        metrics = metrics_summary[table_type]
        print("Results on {} tables ({} total):".format(table_type, metrics['num_tables']))
        print("      Accuracy_Con: {:.4f}".format(metrics['acc_con']))
        print("         GriTS_Top: {:.4f}".format(metrics['grits_top']))
        print("         GriTS_Con: {:.4f}".format(metrics['grits_con']))
        print("         GriTS_Loc: {:.4f}".format(metrics['grits_loc']))
        if 'grits_rawloc' in metrics:
            print("      GriTS_RawLoc: {:.4f}".format(metrics['grits_rawloc']))
        if 'dar_con_original' in metrics:
            print("DAR_Con (original): {:.4f}".format(metrics['dar_con_original']))
        if 'dar_con' in metrics:
            print("           DAR_Con: {:.4f}".format(metrics['dar_con']))
        print('-' * 50)


def eval_tsr(args, model, dataset_test, device):
    """
    Compute table structure recognition (TSR) metrics, including
    grid table similarity (GriTS) and directed adjacency relations (DAR).
    """
    structure_class_names = [
        'table', 'table column', 'table row', 'table column header',
        'table projected row header', 'table spanning cell', 'no object'
    ]
    structure_class_map = {k: v for v, k in enumerate(structure_class_names)}
    structure_class_thresholds = {
        "table": 0.5,
        "table column": 0.5,
        "table row": 0.5,
        "table column header": 0.5,
        "table projected row header": 0.5,
        "table spanning cell": 0.5,
        "no object": 10
    }

    if args.debug:
        max_samples = min(50, len(dataset_test))
    else:
        max_samples = len(dataset_test)
    print("Evaluating {} samples...".format(max_samples))

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model.eval()
    all_metrics = []
    st_time = datetime.now()

    for idx in range(0, max_samples):
        print(idx, end='\r')

        #---Read source data: image, objects, and word bounding boxes
        img, gt, orig_img, img_path = dataset_test[idx]
        img_filename = img_path.split("/")[-1]
        img_words_filepath = os.path.join(args.table_words_dir, img_filename.replace(".jpg", "_words.json"))
        with open(img_words_filepath, 'r') as f:
            page_tokens = json.load(f)
        img_test = img
        scale = 1000 / max(orig_img.size)
        img = normalize(img)
        for word in page_tokens:
            word['bbox'] = [elem * scale for elem in word['bbox']]

        #---Compute ground truth features
        true_bboxes = [list(elem) for elem in gt['boxes'].cpu().numpy()]
        true_labels = gt['labels'].cpu().numpy()
        true_scores = [1 for elem in true_bboxes]
        true_table_structures, true_cells, _ = objects_to_cells(true_bboxes, true_labels, true_scores,
                                                                page_tokens, structure_class_names,
                                                                structure_class_thresholds, structure_class_map)

        #---Compute predicted features
        # Propagate through the model
        with torch.no_grad():
            outputs = model([img.to(device)])
        boxes = outputs['pred_boxes']
        m = outputs['pred_logits'].softmax(-1).max(-1)
        scores = m.values
        labels = m.indices
        #rescaled_bboxes = rescale_bboxes(torch.tensor(boxes[0], dtype=torch.float32), img_test.size)
        rescaled_bboxes = rescale_bboxes(boxes[0].cpu(), img_test.size)
        pred_bboxes = [bbox.tolist() for bbox in rescaled_bboxes]
        pred_labels = labels[0].tolist()
        pred_scores = scores[0].tolist()
        _, pred_cells, _ = objects_to_cells(pred_bboxes, pred_labels, pred_scores,
                                            page_tokens, structure_class_names,
                                            structure_class_thresholds, structure_class_map)

        metrics = compute_metrics(args.mode, true_bboxes, true_labels, true_scores, true_cells,
                                  pred_bboxes, pred_labels, pred_scores, pred_cells)
        statistics = compute_statistics(true_table_structures, true_cells)

        metrics.update(statistics)
        metrics['id'] = img_path.split('/')[-1].split('.')[0]
        all_metrics.append(metrics)

        #---Display output for debugging
        if args.debug:
            print("Sample {}:".format(idx+1))
            print("                 GriTS_Loc: {:.4f}".format(metrics["grits_loc"]))
            print("                 GriTS_Con: {:.4f}".format(metrics["grits_con"]))
            print("                 GriTS_Top: {:.4f}".format(metrics["grits_top"]))
            if args.mode == 'grits-all':
                print("              GriTS_RawLoc: {:.4f}".format(metrics["grits_rawloc"]))
                print("DAR_Con (original version): {:.4f}".format(metrics["dar_original_con"]))
                print("                   DAR_Con: {:.4f}".format(metrics["dar_con"]))

            fig,ax = plt.subplots(1)
            ax.imshow(img_test, interpolation='lanczos')
            fig.set_size_inches((15, 18))
            plt.show()

            fig,ax = plt.subplots(1)
            ax.imshow(img_test, interpolation='lanczos')

            linewidth = 1
            alpha = 0
            for word in page_tokens:
                bbox = word['bbox']
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                         edgecolor='none',facecolor="orange", alpha=0.04)
                ax.add_patch(rect)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                         edgecolor="orange",facecolor='none',linestyle="--")
                ax.add_patch(rect)         
            rescaled_bboxes = rescale_bboxes(boxes[0].cpu(), img_test.size)
            for bbox, label, score in zip(rescaled_bboxes, labels[0].tolist(), scores[0].tolist()):
                bbox = bbox.cpu().numpy().tolist()
                if not label > 5 and score > 0.5:
                    color, alpha, linewidth = get_bbox_decorations(label, score)
                    rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                             edgecolor='none',facecolor=color, alpha=alpha)
                    ax.add_patch(rect)
                    rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                             edgecolor=color,facecolor='none',linestyle="--")
                    ax.add_patch(rect) 

            fig.set_size_inches((15, 18))
            plt.show()

            fig,ax = plt.subplots(1)
            ax.imshow(img_test, interpolation='lanczos')    
            for cell in true_cells:
                bbox = cell['bbox']
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                         edgecolor='none',facecolor="brown", alpha=0.04)
                ax.add_patch(rect)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                         edgecolor="brown",facecolor='none',linestyle="--")
                ax.add_patch(rect) 
                cell_rect = Rect()
                for span in cell['spans']:
                    bbox = span['bbox']
                    rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                             edgecolor='none',facecolor="green", alpha=0.2)
                    ax.add_patch(rect)
                    rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                             edgecolor="green",facecolor='none',linestyle="--")
                    ax.add_patch(rect) 
                    cell_rect.includeRect(bbox)
                if cell_rect.getArea() > 0:
                    bbox = list(cell_rect)
                    rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                             edgecolor='none',facecolor="red", alpha=0.15)
                    ax.add_patch(rect)
                    rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                             edgecolor="red",facecolor='none',linestyle="--")
                    ax.add_patch(rect)

            fig.set_size_inches((15, 18))
            plt.show()

            fig,ax = plt.subplots(1)
            ax.imshow(img_test, interpolation='lanczos')

            for cell in pred_cells:
                bbox = cell['bbox']
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                         edgecolor='none',facecolor="magenta", alpha=0.15)
                ax.add_patch(rect)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                         edgecolor="magenta",facecolor='none',linestyle="--")
                ax.add_patch(rect) 
                cell_rect = Rect()
                for span in cell['spans']:
                    bbox = span['bbox']
                    rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                             edgecolor='none',facecolor="green", alpha=0.2)
                    ax.add_patch(rect)
                    rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                             edgecolor="green",facecolor='none',linestyle="--")
                    ax.add_patch(rect) 
                    cell_rect.includeRect(bbox)
                if cell_rect.getArea() > 0:
                    bbox = list(cell_rect)
                    rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                             edgecolor='none',facecolor="red", alpha=0.15)
                    ax.add_patch(rect)
                    rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                             edgecolor="red",facecolor='none',linestyle="--")
                    ax.add_patch(rect) 

            fig.set_size_inches((15, 18))
            plt.show()

        if (idx+1) % 1000 == 0 or (idx+1) == max_samples:
            # Save sample-level metrics for more analysis
            if len(args.metrics_save_filepath) > 0:
                with open(args.metrics_save_filepath, 'w') as outfile:
                    json.dump(all_metrics, outfile)
            print("Total time taken for {} samples: {}".format(idx+1, datetime.now() - st_time))

    # Compute metrics averaged over all samples
    metrics_summary = compute_metrics_summary(all_metrics, args.mode)

    # Print summary of metrics
    print_metrics_summary(metrics_summary)

    # We can plot the graphs to see the correlation between different variations
    # of similarity metrics by using plot_graph fn as shown below
    #
    # plot_graph([result[0] for result in results], [result[2] for result in results], "Raw BBox IoU", "BBox IoU")


def eval_coco(model, criterion, postprocessors, data_loader_test, dataset_test, device):
    """
    Use this function to do COCO evaluation. Default implementation runs it on
    the test set.
    """
    pubmed_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                            data_loader_test, dataset_test,
                                            device, None)
    print("pubmed: AP50: {:.3f}, AP75: {:.3f}, AP: {:.3f}, AR: {:.3f}".format(
        pubmed_stats['coco_eval_bbox'][1], pubmed_stats['coco_eval_bbox'][2],
        pubmed_stats['coco_eval_bbox'][0], pubmed_stats['coco_eval_bbox'][8]))


if __name__ == "__main__":
    main()
