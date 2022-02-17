"""
Copyright (C) 2021 Microsoft Corporation
"""
import os
import argparse
import sys
import random
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import itertools
import math
import json
import traceback
import statistics as stat
from tqdm import tqdm
from datetime import datetime
from difflib import SequenceMatcher

import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset, ConcatDataset
from torchvision import transforms
import numpy as np
import PIL
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.patches as patches
import fitz
from fitz import Rect

sys.path.append("../detr")
from engine import evaluate
from models import build_model
import util.misc as utils
import datasets.transforms as R
from datasets.coco_eval import CocoEvaluator
import eval_utils
from table_datasets import PDFTablesDataset, RandomMaxResize


def transpose(matrix):
    return list(map(list, zip(*matrix)))


def get_supercell_rows_and_columns(supercells, rows, columns):
    matches_by_supercell = []
    all_matches = set()
    for supercell in supercells:
        row_matches = set()
        column_matches = set()
        for row_num, row in enumerate(rows):
            bbox1 = [
                supercell['bbox'][0], row['bbox'][1], supercell['bbox'][2],
                row['bbox'][3]
            ]
            bbox2 = Rect(supercell['bbox']).intersect(bbox1)
            if bbox2.getArea() / Rect(bbox1).getArea() >= 0.5:
                row_matches.add(row_num)
        for column_num, column in enumerate(columns):
            bbox1 = [
                column['bbox'][0], supercell['bbox'][1], column['bbox'][2],
                supercell['bbox'][3]
            ]
            bbox2 = Rect(supercell['bbox']).intersect(bbox1)
            if bbox2.getArea() / Rect(bbox1).getArea() >= 0.5:
                column_matches.add(column_num)
        already_taken = False
        this_matches = []
        for row_num in row_matches:
            for column_num in column_matches:
                this_matches.append((row_num, column_num))
                if (row_num, column_num) in all_matches:
                    already_taken = True
        if not already_taken:
            for match in this_matches:
                all_matches.add(match)
            matches_by_supercell.append(this_matches)
            row_nums = [elem[0] for elem in this_matches]
            column_nums = [elem[1] for elem in this_matches]
            row_rect = Rect()
            for row_num in row_nums:
                row_rect.includeRect(rows[row_num]['bbox'])
            column_rect = Rect()
            for column_num in column_nums:
                column_rect.includeRect(columns[column_num]['bbox'])
            supercell['bbox'] = list(row_rect.intersect(column_rect))
        else:
            matches_by_supercell.append([])

    return matches_by_supercell


def align_1d(sequence1, sequence2, reward_function, return_alignment=False):
    '''
    Dynamic programming sequence alignment between two sequences
    Traceback convention: -1 = up, 1 = left, 0 = diag up-left
    '''
    sequence1_length = len(sequence1)
    sequence2_length = len(sequence2)
    
    scores = np.zeros((sequence1_length + 1, sequence2_length + 1))
    pointers = np.zeros((sequence1_length + 1, sequence2_length + 1))
    
    # Initialize first column
    for row_idx in range(1, sequence1_length + 1):
        pointers[row_idx, 0] = -1
        
    # Initialize first row
    for col_idx in range(1, sequence2_length + 1):
        pointers[0, col_idx] = 1
        
    for row_idx in range(1, sequence1_length + 1):
        for col_idx in range(1, sequence2_length + 1):
            reward = reward_function(sequence1[row_idx-1], sequence2[col_idx-1])
            diag_score = scores[row_idx - 1, col_idx - 1] + reward
            same_row_score = scores[row_idx, col_idx - 1]
            same_col_score = scores[row_idx - 1, col_idx]
               
            max_score = max(diag_score, same_col_score, same_row_score)
            scores[row_idx, col_idx] = max_score
            if diag_score == max_score:
                pointers[row_idx, col_idx] = 0
            elif same_col_score == max_score:
                pointers[row_idx, col_idx] = -1
            else:
                pointers[row_idx, col_idx] = 1
    
    score = scores[sequence1_length, sequence2_length]
    score = 2 * score / (sequence1_length + sequence2_length)
    
    if not return_alignment:
        return score
    
    # Backtrace
    cur_row = sequence1_length
    cur_col = sequence2_length
    aligned_sequence1_indices = []
    aligned_sequence2_indices = []
    while not (cur_row == 0 and cur_col == 0):
        if pointers[cur_row, cur_col] == -1:
            cur_row -= 1
        elif pointers[cur_row, cur_col] == 1:
            cur_col -= 1
        else:
            cur_row -= 1
            cur_col -= 1
            aligned_sequence1_indices.append(cur_col)
            aligned_sequence2_indices.append(cur_row)
            
    aligned_sequence1_indices = aligned_sequence1_indices[::-1]
    aligned_sequence2_indices = aligned_sequence2_indices[::-1]
    
    return aligned_sequence1_indices, aligned_sequence2_indices, score


def objects_to_cells(bboxes, labels, scores, page_tokens, structure_class_names, structure_class_thresholds, structure_class_map):
    bboxes, scores, labels = eval_utils.apply_class_thresholds(bboxes, labels, scores,
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
    
    tokens_in_table = [token for token in page_tokens if eval_utils.iob(token['bbox'], table_bbox) >= 0.5]
    
    # Determine the table cell structure from the objects
    table_structures, cells, confidence_score = eval_utils.objects_to_cells(table, table_objects, tokens_in_table,
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


def adjacency_metrics(true_adjacencies, pred_adjacencies):
    true_c = Counter()
    true_c.update([elem for elem in true_adjacencies])

    pred_c = Counter()
    pred_c.update([elem for elem in pred_adjacencies])

    if len(true_adjacencies) > 0:
        recall = (sum(true_c.values()) - sum((true_c - pred_c).values())) / sum(true_c.values())
    else:
        recall = 1
    if len(pred_adjacencies) > 0:
        precision = (sum(pred_c.values()) - sum((pred_c - true_c).values())) / sum(pred_c.values())
    else:
        precision = 1

    if recall + precision == 0:
        f_score = 0
    else:
        f_score = 2 * recall * precision / (recall + precision)

    return recall, precision, f_score


def adjacency_metric(true_cells, pred_cells):
    true_adjacencies, _ = cells_to_adjacency_pair_list(true_cells)
    pred_adjacencies, _ = cells_to_adjacency_pair_list(pred_cells)

    return adjacency_metrics(true_adjacencies, pred_adjacencies)


def adjacency_with_blanks_metric(true_cells, pred_cells):
    true_adjacencies, _ = cells_to_adjacency_pair_list_with_blanks(true_cells)
    pred_adjacencies, _ = cells_to_adjacency_pair_list_with_blanks(pred_cells)

    return adjacency_metrics(true_adjacencies, pred_adjacencies)


def cells_to_grid(cells, key='bbox'):
    if len(cells) == 0:
        return [[]]
    num_rows = max([max(cell['row_nums']) for cell in cells])+1
    num_columns = max([max(cell['column_nums']) for cell in cells])+1
    cell_grid = np.zeros((num_rows, num_columns)).tolist()
    for cell in cells:
        for row_num in cell['row_nums']:
            for column_num in cell['column_nums']:
                cell_grid[row_num][column_num] = cell[key]
                
    return cell_grid


def cells_to_relspan_grid(cells):
    if len(cells) == 0:
        return [[]]
    num_rows = max([max(cell['row_nums']) for cell in cells])+1
    num_columns = max([max(cell['column_nums']) for cell in cells])+1
    cell_grid = np.zeros((num_rows, num_columns)).tolist()
    for cell in cells:
        min_row_num = min(cell['row_nums'])
        min_column_num = min(cell['column_nums'])
        max_row_num = max(cell['row_nums']) + 1
        max_column_num = max(cell['column_nums']) + 1
        for row_num in cell['row_nums']:
            for column_num in cell['column_nums']:
                cell_grid[row_num][column_num] = [
                    min_column_num - column_num,
                    min_row_num - row_num,
                    max_column_num - column_num,
                    max_row_num - row_num, 
                ]
                
    return cell_grid


def align_cells_outer(true_cells, pred_cells, reward_function):
    '''
    Dynamic programming sequence alignment between two sequences
    Traceback convention: -1 = up, 1 = left, 0 = diag up-left
    '''
    
    scores = np.zeros((len(true_cells) + 1, len(pred_cells) + 1))
    pointers = np.zeros((len(true_cells) + 1, len(pred_cells) + 1))
    
    # Initialize first column
    for row_idx in range(1, len(true_cells) + 1):
        pointers[row_idx, 0] = -1
        
    # Initialize first row
    for col_idx in range(1, len(pred_cells) + 1):
        pointers[0, col_idx] = 1
        
    for row_idx in range(1, len(true_cells) + 1):
        for col_idx in range(1, len(pred_cells) + 1):
            reward = align_1d(true_cells[row_idx-1], pred_cells[col_idx-1], reward_function)
            diag_score = scores[row_idx - 1, col_idx - 1] + reward
            same_row_score = scores[row_idx, col_idx - 1]
            same_col_score = scores[row_idx - 1, col_idx]
               
            max_score = max(diag_score, same_col_score, same_row_score)
            scores[row_idx, col_idx] = max_score
            if diag_score == max_score:
                pointers[row_idx, col_idx] = 0
            elif same_col_score == max_score:
                pointers[row_idx, col_idx] = -1
            else:
                pointers[row_idx, col_idx] = 1
    
    score = scores[len(true_cells), len(pred_cells)]
    if len(pred_cells) > 0:
        precision = score / len(pred_cells)
    else:
        precision = 1
    if len(true_cells) > 0:
        recall = score / len(true_cells)
    else:
        recall = 1
    score = 2 * precision * recall / (precision + recall)
    #score = 2 * score / (len(true_cells) + len(pred_cells))
    
    cur_row = len(true_cells)
    cur_col = len(pred_cells)
    aligned_true_indices = []
    aligned_pred_indices = []
    while not (cur_row == 0 and cur_col == 0):
        if pointers[cur_row, cur_col] == -1:
            cur_row -= 1
        elif pointers[cur_row, cur_col] == 1:
            cur_col -= 1
        else:
            cur_row -= 1
            cur_col -= 1
            aligned_pred_indices.append(cur_col)
            aligned_true_indices.append(cur_row)
            
    aligned_true_indices = aligned_true_indices[::-1]
    aligned_pred_indices = aligned_pred_indices[::-1]
    
    return aligned_true_indices, aligned_pred_indices, score


def factored_2dlcs(true_cell_grid, pred_cell_grid, reward_function):
    true_row_nums, pred_row_nums, row_score = align_cells_outer(true_cell_grid,
                                                                pred_cell_grid,
                                                                reward_function)
    true_column_nums, pred_column_nums, column_score = align_cells_outer(transpose(true_cell_grid),
                                                                         transpose(pred_cell_grid),
                                                                         reward_function)

    score = 0
    for true_row_num, pred_row_num in zip(true_row_nums, pred_row_nums):
        for true_column_num, pred_column_num in zip(true_column_nums, pred_column_nums):
            score += reward_function(true_cell_grid[true_row_num][true_column_num],
                                     pred_cell_grid[pred_row_num][pred_column_num])

    if true_cell_grid.shape[0] > 0 and true_cell_grid.shape[1] > 0:
        recall = score / (true_cell_grid.shape[0]*true_cell_grid.shape[1])
    else:
        recall = 1
    if pred_cell_grid.shape[0] > 0 and pred_cell_grid.shape[1] > 0:
        precision = score / (pred_cell_grid.shape[0]*pred_cell_grid.shape[1])
    else:
        precision = 1

    if precision > 0 and recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0
    
    return fscore, precision, recall, row_score, column_score


def lcs_similarity(string1, string2):
    if len(string1) == 0 and len(string2) == 0:
        return 1
    s = SequenceMatcher(None, string1, string2)
    lcs = ''.join([string1[block.a:(block.a + block.size)] for block in s.get_matching_blocks()])
    return 2*len(lcs)/(len(string1)+len(string2))


def output_to_dilatedbbox_grid(bboxes, labels, scores):
    rows = [{'bbox': bbox} for bbox, label in zip(bboxes, labels) if label == 2]
    columns = [{'bbox': bbox} for bbox, label in zip(bboxes, labels) if label == 1]
    supercells = [{'bbox': bbox, 'score': 1} for bbox, label in zip(bboxes, labels) if label in [4, 5]]
    rows.sort(key=lambda x: x['bbox'][1]+x['bbox'][3])
    columns.sort(key=lambda x: x['bbox'][0]+x['bbox'][2])
    supercells.sort(key=lambda x: -x['score'])
    cell_grid = []
    for row_num, row in enumerate(rows):
        column_grid = []
        for column_num, column in enumerate(columns):
            bbox = Rect(row['bbox']).intersect(column['bbox'])
            column_grid.append(list(bbox))
        cell_grid.append(column_grid)
    matches_by_supercell = get_supercell_rows_and_columns(supercells, rows, columns)
    for matches, supercell in zip(matches_by_supercell, supercells):
        for match in matches:
            cell_grid[match[0]][match[1]] = supercell['bbox']
    
    return cell_grid


def compute_metrics(true_bboxes, true_labels, true_scores, true_cells,
                    pred_bboxes, pred_labels, pred_scores, pred_cells):

    # Compute grids/matrices for comparison
    true_cell_dilatedbbox_grid = np.array(output_to_dilatedbbox_grid(true_bboxes, true_labels, true_scores))
    true_relspan_grid = np.array(cells_to_relspan_grid(true_cells))
    true_bbox_grid = np.array(cells_to_grid(true_cells, key='bbox'))
    true_text_grid = np.array(cells_to_grid(true_cells, key='cell_text'), dtype=object)

    pred_cell_dilatedbbox_grid = np.array(output_to_dilatedbbox_grid(pred_bboxes, pred_labels, pred_scores))
    pred_relspan_grid = np.array(cells_to_relspan_grid(pred_cells))
    pred_bbox_grid = np.array(cells_to_grid(pred_cells, key='bbox'))
    pred_text_grid = np.array(cells_to_grid(pred_cells, key='cell_text'), dtype=object)

    #---Compute each of the metrics
    metrics = {}
    (metrics['grits_rawloc'], metrics['grits_precision_rawloc'],
     metrics['grits_recall_rawloc'], metrics['grits_rawloc_rowbased'],
     metrics['grits_rawloc_columnbased']) = factored_2dlcs(true_cell_dilatedbbox_grid,
                                                pred_cell_dilatedbbox_grid,
                                                reward_function=eval_utils.iou)

    (metrics['grits_top'], metrics['grits_precision_top'],
     metrics['grits_recall_top'], metrics['grits_top_rowbased'],
     metrics['grits_top_columnbased']) = factored_2dlcs(true_relspan_grid,
                                             pred_relspan_grid,
                                             reward_function=eval_utils.iou)

    (metrics['grits_loc'], metrics['grits_precision_loc'],
     metrics['grits_recall_loc'], metrics['grits_loc_rowbased'],
     metrics['grits_loc_columnbased']) = factored_2dlcs(true_bbox_grid,
                                             pred_bbox_grid,
                                             reward_function=eval_utils.iou)

    (metrics['grits_cont'], metrics['grits_precision_cont'],
     metrics['grits_recall_cont'], metrics['grits_cont_rowbased'],
     metrics['grits_cont_columnbased']) = factored_2dlcs(true_text_grid,
                                             pred_text_grid,
                                             reward_function=lcs_similarity)

    (metrics['adjacency_nonblank_recall'], metrics['adjacency_nonblank_precision'],
     metrics['adjacency_nonblank_fscore']) = adjacency_metric(true_cells, pred_cells)

    (metrics['adjacency_withblank_recall'], metrics['adjacency_withblank_precision'],
     metrics['adjacency_withblank_fscore']) = adjacency_with_blanks_metric(true_cells, pred_cells)

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


def grits(args, model, dataset_test, device):
    """
    This function runs the GriTS proposed in the paper. We also have a debug
    mode which let's you see the outputs of a model on the pdf pages.
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
        max_samples = 50
    else:
        max_samples = len(dataset_test)
    print(max_samples)

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
        true_table_structures, true_cells, true_confidence_score = objects_to_cells(true_bboxes, true_labels, true_scores,
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
        pred_table_structures, pred_cells, pred_confidence_score = objects_to_cells(pred_bboxes, pred_labels, pred_scores,
                                                                                    page_tokens, structure_class_names,
                                                                                    structure_class_thresholds, structure_class_map)

        metrics = compute_metrics(true_bboxes, true_labels, true_scores, true_cells,
                                  pred_bboxes, pred_labels, pred_scores, pred_cells)
        statistics = compute_statistics(true_table_structures, true_cells)

        metrics.update(statistics)
        metrics['id'] = img_path.split('/')[-1].split('.')[0]
        all_metrics.append(metrics)

        if idx%1000==0:
            with open(args.metrics_save_filepath, 'w') as outfile:
                json.dump(all_metrics, outfile)
            print("Total time taken for {} samples: {}".format(idx, datetime.now() - st_time))

        #---Display output for debugging
        if args.debug:
            print("GriTS RawLoc: {}".format(metrics["grits_rawloc"]))
            print("GriTS Loc: {}".format(metrics["grits_loc"]))
            print("GriTS Top: {}".format(metrics["grits_top"]))
            print("GriTS Cont: {}".format(metrics["grits_cont"]))
            print("Adjacency f-score: {}".format(metrics["adjacency_nonblank_fscore"]))
            print("Adjacency w/ blanks f-score: {}".format(metrics["adjacency_withblank_fscore"]))

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
            rescaled_bboxes = rescale_bboxes(torch.tensor(boxes[0], dtype=torch.float32), img_test.size)
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
    with open(args.metrics_save_filepath, 'w') as outfile:
        json.dump(all_metrics, outfile)
    print("Total time taken: ", datetime.now() - st_time)

    print('-' * 100)
    results = [result for result in all_metrics if result['num_spanning_cells'] == 0]
    print("Results on simple tables ({} total):".format(len(results)))
    print("GriTS_RawLoc: {}".format(np.mean([result['grits_rawloc'] for result in results])))
    print("GriTS_Loc: {}".format(np.mean([result['grits_loc'] for result in results])))
    print("GriTS_Cont: {}".format(np.mean([result['grits_cont'] for result in results])))
    print("GriTS_Top: {}".format(np.mean([result['grits_top'] for result in results])))
    print("Adjacency f-score: {}".format(np.mean([result['adjacency_nonblank_fscore'] for result in results])))
    print("Adjacency w/ blanks f-score: {}".format(np.mean([result['adjacency_withblank_fscore'] for result in results])))

    print('-' * 50)
    results = [result for result in all_metrics if result['num_spanning_cells'] > 0]
    print("Results on complicated tables ({} total):".format(len(results)))
    print("GriTS_RawLoc: {}".format(np.mean([result['grits_rawloc'] for result in results])))
    print("GriTS_Loc: {}".format(np.mean([result['grits_loc'] for result in results])))
    print("GriTS_Cont: {}".format(np.mean([result['grits_cont'] for result in results])))
    print("GriTS_Top: {}".format(np.mean([result['grits_top'] for result in results])))
    print("Adjacency f-score: {}".format(np.mean([result['adjacency_nonblank_fscore'] for result in results])))
    print("Adjacency w/ blanks f-score: {}".format(np.mean([result['adjacency_withblank_fscore'] for result in results])))

    print('-' * 50)
    results = [result for result in all_metrics]
    print("Results on all tables ({} total):".format(len(results)))
    print("GriTS_RawLoc: {}".format(np.mean([result['grits_rawloc'] for result in results if not math.isnan(result['grits_rawloc'])])))
    print("GriTS_Loc: {}".format(np.mean([result['grits_loc'] for result in results])))
    print("GriTS_Cont: {}".format(np.mean([result['grits_cont'] for result in results])))
    print("GriTS_Top: {}".format(np.mean([result['grits_top'] for result in results])))
    print("Adjacency f-score: {}".format(np.mean([result['adjacency_nonblank_fscore'] for result in results])))
    print("Adjacency w/ blanks f-score: {}".format(np.mean([result['adjacency_withblank_fscore'] for result in results])))
    # We can plot the graphs to see the correlation between different variations
    # of similarity metrics by using plot_graph fn as shown below
    #
    # plot_graph([result[0] for result in results], [result[2] for result in results], "Raw BBox IoU", "BBox IoU")


if __name__ == "__main__":
    main()
