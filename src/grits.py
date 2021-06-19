"""
Copyright (C) 2021 Microsoft Corporation
"""
import os
import argparse
import sys
import random
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
import itertools
import math
import json
import traceback
from tqdm import tqdm
from datetime import datetime

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
from config import Args
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
            bbox1 = [supercell['bbox'][0], row['bbox'][1], supercell['bbox'][2], row['bbox'][3]]
            bbox2 = Rect(supercell['bbox']).intersect(bbox1)
            if bbox2.getArea() / Rect(bbox1).getArea() >= 0.5:
                row_matches.add(row_num)
        for column_num, column in enumerate(columns):
            bbox1 = [column['bbox'][0], supercell['bbox'][1], column['bbox'][2], supercell['bbox'][3]]
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
    table_objects = []
    for bbox, score, label in zip(bboxes, scores, labels):
        table_objects.append({'bbox': bbox, 'score': score, 'label': label})
        
    table = {'objects': table_objects, 'page_num': 0} 
    
    table_class_objects = [obj for obj in table_objects if obj['label'] == structure_class_map['table']]
    if len(table_class_objects) > 1:
        table_class_objects = sorted(table_class_objects, key=lambda x: x['score'], reverse=True)
    table_bbox = list(table_class_objects[0]['bbox'])
    table['bbox'] = list(table_bbox)
    
    tokens_in_table = [token for token in page_tokens if eval_utils.iob(token['bbox'], table_bbox) >= 0.5]
    
    # Determine the table cell structure from the objects
    table_structures, cells, confidence_score = eval_utils.objects_to_cells(table, table_objects, tokens_in_table,
                                                                    structure_class_names,
                                                                    structure_class_thresholds)
    
    return table_structures, cells, confidence_score


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
    score = 2 * score / (len(true_cells) + len(pred_cells))
    
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
    
    true_subtable = true_cell_grid[np.ix_(true_row_nums, true_column_nums)].tolist()
    pred_subtable = pred_cell_grid[np.ix_(pred_row_nums, pred_column_nums)].tolist()
    
    num_rows = len(true_row_nums)
    num_columns = len(true_column_nums)

    score = 0
    for idx1 in range(num_rows):
        for idx2 in range(num_columns):
            score += reward_function(true_subtable[idx1][idx2], pred_subtable[idx1][idx2])
    score = 2 * score / (true_cell_grid.shape[0]*true_cell_grid.shape[1] + pred_cell_grid.shape[0]*pred_cell_grid.shape[1])
    
    return score, row_score, column_score


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


def make_align1d_reward_function(reward_function, return_alignment=False):
    def align_1d(sequence1, sequence2):
        '''
        Dynamic programming sequence alignment between two sequences
        Traceback convention: -1 = up, 1 = left, 0 = diag up-left
        '''
        sequence1_length = len(sequence1)
        sequence2_length = len(sequence2)
        
        if sequence1_length == 0 and sequence2_length == 0:
            return 1.0
        elif sequence1_length == 0 or sequence2_length == 0:
            return 0.0
        
        # First see if the sequences are equal:
        is_equal = True
        if sequence1_length == sequence2_length:
            for idx in range(sequence1_length):
                reward = reward_function(sequence1[idx], sequence2[idx])
                if not reward == 1:
                    is_equal = False
                    break      
        if is_equal:
            if not return_alignment:
                return 1.0
            else:
                return list(range(sequence1_length)), list(range(sequence2_length)), 1.0

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

    return align_1d


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b



def get_bbox_decorations(label, score):
    colors = ['brown', 'red', 'blue', 'magenta', 'cyan', 'green', 'orange', 'green', 'orange', 'yellow',
         'brown', 'red', 'blue', 'magenta', 'cyan', 'green', 'orange', 'green', 'orange', 'yellow']
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
            'table', 'table column', 'table row', 'table column header', 'table projected row header', 'table spanning cell', 'no object'
    ]
    structure_class_map = {k: v for v, k in enumerate(structure_class_names)}
    structure_class_thresholds = {
            "table": 0.5, "table column": 0.5, "table row": 0.5, "table column header": 0.5, "table projected row header": 0.5,
            "table spanning cell": 0.5, "no object": 10
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
    
    simple_results = []
    complicated_results = []
    st_time = datetime.now()
    
    for idx in range(max_samples):
        #---Read source data: image, objects, and word bounding boxes
        curr_time = datetime.now()
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
        curr_time = datetime.now()
        
        true_bboxes = [list(elem) for elem in gt['boxes'].cpu().numpy()]
        true_labels = gt['labels'].cpu().numpy()
        true_scores = [1 for elem in true_bboxes]
        true_cell_dilatedbbox_grid = np.array(output_to_dilatedbbox_grid(true_bboxes, true_labels, true_scores))
        true_table_structures, true_cells, true_confidence_score = objects_to_cells(true_bboxes, true_labels, true_scores,
                                                                                    page_tokens, structure_class_names,
                                                                                    structure_class_thresholds,
                                                                                    structure_class_map)
        true_relspan_grid = np.array(cells_to_relspan_grid(true_cells))
        true_bbox_grid = np.array(cells_to_grid(true_cells, key='bbox'))
        true_text_grid = np.array(cells_to_grid(true_cells, key='cell_text'), dtype=object)
        
        #---Compute predicted features
        # Propagate through the model
        curr_time = datetime.now()
        with torch.no_grad():
            outputs = model([img.to(device)])
        boxes = outputs['pred_boxes']
        m = outputs['pred_logits'].softmax(-1).max(-1)
        scores = m.values
        labels = m.indices
        rescaled_bboxes = rescale_bboxes(torch.tensor(boxes[0], dtype=torch.float32), img_test.size)
        pred_bboxes = [bbox.tolist() for bbox in rescaled_bboxes]
        pred_labels = labels[0].tolist()
        pred_scores = scores[0].tolist()
        
        pred_bboxes, pred_scores, pred_labels = eval_utils.apply_class_thresholds(pred_bboxes, pred_labels, pred_scores,
                                                structure_class_names,
                                                structure_class_thresholds)
        pred_cell_dilatedbbox_grid = np.array(output_to_dilatedbbox_grid(pred_bboxes, pred_labels, pred_scores))
        pred_table_structures, pred_cells, pred_confidence_score = objects_to_cells(pred_bboxes, pred_labels, pred_scores,
                                                                                    page_tokens, structure_class_names,
                                                                                    structure_class_thresholds,
                                                                                    structure_class_map)
        pred_relspan_grid = np.array(cells_to_relspan_grid(pred_cells))
        pred_bbox_grid = np.array(cells_to_grid(pred_cells, key='bbox'))
        pred_text_grid = np.array(cells_to_grid(pred_cells, key='cell_text'), dtype=object)
        
        #---Compute each of the metrics
        curr_time = datetime.now()
        combined_dilatedbbox_score, row_dilatedbbox_score, column_dilatedbbox_score = factored_2dlcs(true_cell_dilatedbbox_grid, pred_cell_dilatedbbox_grid, reward_function=eval_utils.iou)
        combined_relspan_score, row_relspan_score, column_relspan_score = factored_2dlcs(true_relspan_grid, pred_relspan_grid, reward_function=eval_utils.iou)
        combined_iou_score, row_iou_score, column_iou_score = factored_2dlcs(true_bbox_grid, pred_bbox_grid, reward_function=eval_utils.iou)
        combined_lcs_score, row_lcs_score, column_lcs_score = factored_2dlcs(true_text_grid, pred_text_grid, reward_function=make_align1d_reward_function(lambda x, y: 1 if x == y else 0))
        
        #---Collect results
        curr_time = datetime.now()
        result = (combined_dilatedbbox_score, combined_relspan_score, combined_iou_score, combined_lcs_score)
        if 4 in true_labels or 5 in true_labels:
            complicated_results.append(result)
        else:
            simple_results.append(result)
        
        #---Display output for debugging
        if args.debug:
            print("TabS-RawIoU: {}; row-first: {}, column-first: {}".format(combined_dilatedbbox_score, row_dilatedbbox_score, column_dilatedbbox_score))
            print("TabS-RelSpan: {}; row-first: {}, column-first: {}".format(combined_relspan_score, row_relspan_score, column_relspan_score))
            print("TabS-IoU: {}; row-first: {}, column-first: {}".format(combined_iou_score, row_iou_score, column_iou_score))
            print("TabS-Text: {}; row-first: {}, column-first: {}".format(combined_lcs_score, row_lcs_score, column_lcs_score))        
            
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
                if not label > 5 and score > 0.3:
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
        if idx%1000 == 0:
            print(idx)
            print(datetime.now() - st_time)
            
    print("Total time taken for evaluation is ", datetime.now() - st_time)
    
    results = complicated_results
    print('-'*100)
    print("Results on complicated tables:")
    print("Raw Cell Bbox IoU: {}".format(np.mean([result[0] for result in results])))
    print("RelSpan IoU: {}".format(np.mean([result[1] for result in results])))
    print("Cell Bbox IoU: {}".format(np.mean([result[2] for result in results])))
    print("Text LCS: {}".format(np.mean([result[3] for result in results])))
    
    
    results = simple_results
    print('-'*100)
    print("Results on simple tables:")
    print("Raw Cell Bbox IoU: {}".format(np.mean([result[0] for result in results])))
    print("RelSpan IoU: {}".format(np.mean([result[1] for result in results])))
    print("Cell Bbox IoU: {}".format(np.mean([result[2] for result in results])))
    print("Text LCS: {}".format(np.mean([result[3] for result in results])))
    
    
    results = simple_results + complicated_results
    print('-'*100)
    print("Results on all tables:")
    print("Raw Cell Bbox IoU: {}".format(np.mean([result[0] for result in results])))
    print("RelSpan IoU: {}".format(np.mean([result[1] for result in results])))
    print("Cell Bbox IoU: {}".format(np.mean([result[2] for result in results])))
    print("Text LCS: {}".format(np.mean([result[3] for result in results])))

    # We can plot the graphs to see the correlation between different variations
    # of similarity metrics by using plot_graph fn as shown below
    #
    # plot_graph([result[0] for result in results], [result[2] for result in results], "Raw BBox IoU", "BBox IoU")

if __name__ =="__main__":
    main()
