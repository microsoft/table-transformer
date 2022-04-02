"""
Copyright (C) 2021 Microsoft Corporation
"""
import itertools
from difflib import SequenceMatcher

import numpy as np
from fitz import Rect


def compute_fscore(num_true_positives, num_true, num_positives):
    """
    Compute the f-score or f-measure for a collection of predictions.

    Conventions:
    - precision is 1 when there are no predicted instances
    - recall is 1 when there are no true instances
    - fscore is 0 when recall or precision is 0
    """
    if num_positives > 0:
        precision = num_true_positives / num_positives
    else:
        precision = 1
    if num_true > 0:
        recall = num_true_positives / num_true
    else:
        recall = 1
        
    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0

    return fscore, precision, recall 


def initialize_DP(sequence1_length, sequence2_length):
    # Initialize DP tables
    scores = np.zeros((sequence1_length + 1, sequence2_length + 1))
    pointers = np.zeros((sequence1_length + 1, sequence2_length + 1))
    
    # Initialize pointers in DP table
    for seq1_idx in range(1, sequence1_length + 1):
        pointers[seq1_idx, 0] = -1
        
    # Initialize pointers in DP table
    for seq2_idx in range(1, sequence2_length + 1):
        pointers[0, seq2_idx] = 1

    return scores, pointers


def traceback(pointers):
    """
    Dynamic programming traceback to determine the aligned indices
    between the two sequences.

    Traceback convention: -1 = up, 1 = left, 0 = diag up-left
    """
    seq1_idx = pointers.shape[0] - 1
    seq2_idx = pointers.shape[1] - 1
    aligned_sequence1_indices = []
    aligned_sequence2_indices = []
    while not (seq1_idx == 0 and seq2_idx == 0):
        if pointers[seq1_idx, seq2_idx] == -1:
            seq1_idx -= 1
        elif pointers[seq1_idx, seq2_idx] == 1:
            seq2_idx -= 1
        else:
            seq1_idx -= 1
            seq2_idx -= 1
            aligned_sequence1_indices.append(seq1_idx)
            aligned_sequence2_indices.append(seq2_idx)
            
    aligned_sequence1_indices = aligned_sequence1_indices[::-1]
    aligned_sequence2_indices = aligned_sequence2_indices[::-1]

    return aligned_sequence1_indices, aligned_sequence2_indices


def align_1d(sequence1, sequence2, reward_lookup, return_alignment=False):
    '''
    Dynamic programming alignment between two sequences,
    with memoized rewards.

    Sequences are represented as indices into the rewards lookup table.

    Traceback convention: -1 = up, 1 = left, 0 = diag up-left
    '''
    sequence1_length = len(sequence1)
    sequence2_length = len(sequence2)

    scores, pointers = initialize_DP(sequence1_length,
                                     sequence2_length)
        
    for seq1_idx in range(1, sequence1_length+1):
        for seq2_idx in range(1, sequence2_length+1):
            reward = reward_lookup[sequence1[seq1_idx-1] + sequence2[seq2_idx-1]]
            diag_score = scores[seq1_idx-1, seq2_idx-1] + reward
            skip_seq2_score = scores[seq1_idx, seq2_idx-1]
            skip_seq1_score = scores[seq1_idx-1, seq2_idx]
               
            max_score = max(diag_score, skip_seq1_score, skip_seq2_score)
            scores[seq1_idx, seq2_idx] = max_score
            if diag_score == max_score:
                pointers[seq1_idx, seq2_idx] = 0
            elif skip_seq1_score == max_score:
                pointers[seq1_idx, seq2_idx] = -1
            else: # skip_seq2_score == max_score
                pointers[seq1_idx, seq2_idx] = 1
    
    score = scores[-1, -1]
    
    if not return_alignment:
        return score
    
    # Traceback
    sequence1_indices, sequence2_indices = traceback(pointers)
    
    return sequence1_indices, sequence2_indices, score


def align_2d_outer(true_shape, pred_shape, reward_lookup):
    '''
    Dynamic programming matrix alignment posed as 2D
    sequence-of-sequences alignment:
    Align two outer sequences whose entries are also sequences,
    where the match reward between the inner sequence entries
    is their 1D sequence alignment score.

    Traceback convention: -1 = up, 1 = left, 0 = diag up-left
    '''
    
    scores, pointers = initialize_DP(true_shape[0], pred_shape[0])
        
    for row_idx in range(1, true_shape[0] + 1):
        for col_idx in range(1, pred_shape[0] + 1):
            reward = align_1d([(row_idx-1, tcol) for tcol in range(true_shape[1])],
                              [(col_idx-1, prow) for prow in range(pred_shape[1])],
                              reward_lookup)
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
    
    score = scores[-1, -1]
            
    aligned_true_indices, aligned_pred_indices = traceback(pointers)
    
    return aligned_true_indices, aligned_pred_indices, score


def factored_2dmss(true_cell_grid, pred_cell_grid, reward_function):
    """
    Factored 2D-MSS: Factored two-dimensional most-similar substructures

    This is a polynomial-time heuristic to computing the 2D-MSS of two matrices,
    which is NP hard.

    A substructure of a matrix is a subset of its rows and its columns.

    The most similar substructures of two matrices, A and B, are the substructures
    A' and B', where the sum of the similarity over all corresponding entries
    A'(i, j) and B'(i, j) is greatest.
    """
    pre_computed_rewards = {}
    transpose_rewards = {}
    for trow, tcol, prow, pcol in itertools.product(range(true_cell_grid.shape[0]),
                                                    range(true_cell_grid.shape[1]),
                                                    range(pred_cell_grid.shape[0]),
                                                    range(pred_cell_grid.shape[1])):

        reward = reward_function(true_cell_grid[trow, tcol], pred_cell_grid[prow, pcol])

        pre_computed_rewards[(trow, tcol, prow, pcol)] = reward
        transpose_rewards[(tcol, trow, pcol, prow)] = reward

    num_pos = pred_cell_grid.shape[0] * pred_cell_grid.shape[1]
    num_true = true_cell_grid.shape[0] * true_cell_grid.shape[1]

    true_row_nums, pred_row_nums, row_pos_match_score = align_2d_outer(true_cell_grid.shape[:2],
                                                                pred_cell_grid.shape[:2],
                                                                pre_computed_rewards)

    true_column_nums, pred_column_nums, col_pos_match_score = align_2d_outer(true_cell_grid.shape[:2][::-1],
                                                                         pred_cell_grid.shape[:2][::-1],
                                                                         transpose_rewards)

    pos_match_score_upper_bound =  min(row_pos_match_score, col_pos_match_score)
    upper_bound_score, _, _ = compute_fscore(pos_match_score_upper_bound, num_pos, num_true)

    positive_match_score = 0
    for true_row_num, pred_row_num in zip(true_row_nums, pred_row_nums):
        for true_column_num, pred_column_num in zip(true_column_nums, pred_column_nums):
            positive_match_score += pre_computed_rewards[(true_row_num, true_column_num, pred_row_num, pred_column_num)]

    fscore, precision, recall = compute_fscore(positive_match_score,
                                               num_true,
                                               num_pos)
    
    return fscore, precision, recall, upper_bound_score


def lcs_similarity(string1, string2):
    if len(string1) == 0 and len(string2) == 0:
        return 1
    s = SequenceMatcher(None, string1, string2)
    lcs = ''.join([string1[block.a:(block.a + block.size)] for block in s.get_matching_blocks()])
    return 2*len(lcs)/(len(string1)+len(string2))


def iou(bbox1, bbox2):
    """
    Compute the intersection-over-union of two bounding boxes.
    """
    intersection = Rect(bbox1).intersect(bbox2)
    union = Rect(bbox1).includeRect(bbox2)
    
    union_area = union.getArea()
    if union_area > 0:
        return intersection.getArea() / union.getArea()
    
    return 0


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


def grits_top(true_relative_span_grid, pred_relative_span_grid):
    """
    Compute GriTS_Top given two matrices of cell relative spans.

    For the cell at grid location (i,j), let a(i,j) be its rowspan,
    let β(i,j) be its colspan, let p(i,j) be the minimum row it occupies,
    and let θ(i,j) be the minimum column it occupies. Its relative span is
    bounding box [θ(i,j)-j, p(i,j)-i, θ(i,j)-j+β(i,j), p(i,j)-i+a(i,j)].

    It gives the size and location of the cell each grid cell belongs to
    relative to the current grid cell location, in grid coordinate units.
    Note that for a non-spanning cell this will always be [0, 0, 1, 1].
    """
    return factored_2dmss(true_relative_span_grid,
                          pred_relative_span_grid,
                          reward_function=iou)


def grits_loc(true_bbox_grid, pred_bbox_grid):
    """
    Compute GriTS_Loc given two matrices of cell bounding boxes.
    """
    return factored_2dmss(true_bbox_grid,
                          pred_bbox_grid,
                          reward_function=iou)


def grits_con(true_text_grid, pred_text_grid):
    """
    Compute GriTS_Con given two matrices of cell text strings.
    """
    return factored_2dmss(true_text_grid,
                          pred_text_grid,
                          reward_function=lcs_similarity)