"""
Copyright (C) 2021 Microsoft Corporation
"""
import itertools

import numpy as np

import eval_utils
from eval_utils import compute_fscore


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
                          reward_function=eval_utils.iou)


def grits_loc(true_bbox_grid, pred_bbox_grid):
    """
    Compute GriTS_Loc given two matrices of cell bounding boxes.
    """
    return factored_2dmss(true_bbox_grid,
                          pred_bbox_grid,
                          reward_function=eval_utils.iou)


def grits_con(true_text_grid, pred_text_grid):
    """
    Compute GriTS_Con given two matrices of cell text strings.
    """
    return factored_2dmss(true_text_grid,
                          pred_text_grid,
                          reward_function=eval_utils.lcs_similarity)