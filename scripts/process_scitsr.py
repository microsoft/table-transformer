"""
Copyright (C) 2023 Microsoft Corporation

Script to process, edit, filter, and canonicalize SciTSR to align it with PubTables-1M.

We still need to verify that this script works correctly.

If you use this code in your published work, we request that you cite our papers
and table-transformer GitHub repo.
"""

import json
import os
from collections import defaultdict
import traceback
from difflib import SequenceMatcher
import argparse

import fitz
from fitz import Rect
from PIL import Image
import xml.etree.ElementTree as ET
from xml.dom import minidom
import editdistance
import numpy as np
from tqdm import tqdm


def adjust_bbox_coordinates(data, doc):
    # Change bbox coordinates to be relative to PyMuPDF page.rect coordinate space
    media_box = doc[0].mediabox
    mat = doc[0].transformation_matrix

    for cell in ['cells']:
        if not 'bbox' in cell:
            continue
        bbox = list(Rect(cell['bbox']) * mat)
        bbox = [bbox[0] + media_box[0],
                bbox[1] - media_box[1],
                bbox[2] + media_box[0],
                bbox[3] - media_box[1]]
        cell['bbox'] = bbox

def table_to_text(table_dict):
    return ' '.join([cell['text_content'].strip() for cell in  table_dict['cells']])

def align(page_string="", xml_string="", page_character_rewards=None, xml_character_rewards=None, match_reward=2,
          space_match_reward=3, lowercase_match_reward=2, mismatch_penalty=-5,
          page_new_gap_penalty=-2, xml_new_gap_penalty=-5, page_continue_gap_penalty=-0.01, xml_continue_gap_penalty=-0.1,
          page_boundary_gap_reward=0.01, gap_not_after_space_penalty=-1,
          score_only=False, gap_character='_'):
    '''
    Dynamic programming sequence alignment between two text strings; the first text string
    is considered to come from the PDF document; the second text string is considered to
    come from the XML document.
    Traceback convention: -1 = up, 1 = left, 0 = diag up-left
    '''
    
    scores = np.zeros((len(page_string) + 1, len(xml_string) + 1))
    pointers = np.zeros((len(page_string) + 1, len(xml_string) + 1))
    
    # Initialize first column
    for row_idx in range(1, len(page_string) + 1):
        scores[row_idx, 0] = scores[row_idx - 1, 0] + page_boundary_gap_reward
        pointers[row_idx, 0] = -1
        
    # Initialize first row
    for col_idx in range(1, len(xml_string) + 1):
        #scores[0, col_idx] = scores[0, col_idx - 1] + 0
        pointers[0, col_idx] = 1
        
    for row_idx in range(1, len(page_string) + 1):
        for col_idx in range(1, len(xml_string) + 1):
            # Score if matching the characters
            if page_string[row_idx - 1].lower() == xml_string[col_idx - 1].lower():
                if page_string[row_idx - 1] == ' ':
                    reward = space_match_reward
                elif page_string[row_idx - 1] == xml_string[col_idx - 1]:
                    reward = match_reward
                else:
                    reward = lowercase_match_reward
                if not page_character_rewards is None:
                    reward *= page_character_rewards[row_idx-1]
                if not xml_character_rewards is None:
                    reward *= xml_character_rewards[col_idx-1]
                diag_score = scores[row_idx - 1, col_idx - 1] + reward
            else:
                diag_score = scores[row_idx - 1, col_idx - 1] + mismatch_penalty
            
            if pointers[row_idx, col_idx - 1] == 1:
                same_row_score = scores[row_idx, col_idx - 1] + page_continue_gap_penalty
            else:
                same_row_score = scores[row_idx, col_idx - 1] + page_new_gap_penalty
                if not xml_string[col_idx - 1] == ' ':
                    same_row_score += gap_not_after_space_penalty
            
            if col_idx == len(xml_string):
                same_col_score = scores[row_idx - 1, col_idx] + page_boundary_gap_reward
            elif pointers[row_idx - 1, col_idx] == -1:
                same_col_score = scores[row_idx - 1, col_idx] + xml_continue_gap_penalty
            else:
                same_col_score = scores[row_idx - 1, col_idx] + xml_new_gap_penalty
                if not page_string[row_idx - 1] == ' ':
                    same_col_score += gap_not_after_space_penalty
               
            max_score = max(diag_score, same_col_score, same_row_score)
            scores[row_idx, col_idx] = max_score
            if diag_score == max_score:
                pointers[row_idx, col_idx] = 0
            elif same_col_score == max_score:
                pointers[row_idx, col_idx] = -1
            else:
                pointers[row_idx, col_idx] = 1
    
    score = scores[len(page_string), len(xml_string)]
    
    if score_only:
        return score
    
    cur_row = len(page_string)
    cur_col = len(xml_string)
    aligned_page_string = ""
    aligned_xml_string = ""
    while not (cur_row == 0 and cur_col == 0):
        if pointers[cur_row, cur_col] == -1:
            cur_row -= 1
            aligned_xml_string += gap_character
            aligned_page_string += page_string[cur_row]
        elif pointers[cur_row, cur_col] == 1:
            cur_col -= 1
            aligned_page_string += gap_character
            aligned_xml_string += xml_string[cur_col]
        else:
            cur_row -= 1
            cur_col -= 1
            aligned_xml_string += xml_string[cur_col]
            aligned_page_string += page_string[cur_row]
            
    aligned_page_string = aligned_page_string[::-1]
    aligned_xml_string = aligned_xml_string[::-1]
    
    alignment = [aligned_page_string, aligned_xml_string]
    
    return alignment, score


def locate_table(page_words, table):
    #sorted_words = sorted(words, key=functools.cmp_to_key(compare_meta))
    sorted_words = page_words
    page_text = " ".join([word[4] for word in sorted_words])

    page_text_source = []
    for num, word in enumerate(sorted_words):
        for c in word[4]:
            page_text_source.append(num)
        page_text_source.append(None)
    page_text_source = page_text_source[:-1]
        
    table_text = table_to_text(table)
    table_text_source = []
    for num, cell in enumerate(table['cells']):
        for c in cell['text_content'].strip():
            table_text_source.append(num)
        table_text_source.append(None)
    table_text_source = table_text_source[:-1]

    X = page_text.replace("~", "^")
    Y = table_text.replace("~", "^")

    match_reward = 3
    mismatch_penalty = -2
    #new_gap_penalty = -10
    continue_gap_penalty = -0.05
    page_boundary_gap_reward = 0.2

    alignment, score = align(X, Y, match_reward=match_reward, mismatch_penalty=mismatch_penalty,
                             page_boundary_gap_reward=page_boundary_gap_reward, score_only=False,
          gap_character='~')
    
    table_words = set()
    column_words = dict()
    row_words = dict()
    cell_words = dict()
    page_count = 0
    table_count = 0
    
    for char1, char2 in zip(alignment[0], alignment[1]):
        if not char1 == "~":
            if char1 == char2:
                table_words.add(page_text_source[page_count])
                cell_num = table_text_source[table_count]
                if not cell_num is None:
                    if cell_num in cell_words:
                        cell_words[cell_num].add(page_text_source[page_count])
                    else:
                        cell_words[cell_num] = set([page_text_source[page_count]])
            page_count += 1
        if not char2 == "~":
            table_count += 1
            
    inliers = []
    for word_num in table_words:
        if word_num:
            inliers.append(sorted_words[word_num])
    
    if len(inliers) == 0:
        return None, None
        
    cell_bboxes = {}
    for cell_num, cell in enumerate(table['cells']):
        cell_bbox = None
        if cell_num in cell_words:
            for word_num in cell_words[cell_num]:
                if not word_num is None:
                    word_bbox = sorted_words[word_num][0:4]
                    if not cell_bbox:
                        cell_bbox = [entry for entry in word_bbox]
                    else:
                        cell_bbox[0] = min(cell_bbox[0], word_bbox[0])
                        cell_bbox[1] = min(cell_bbox[1], word_bbox[1])
                        cell_bbox[2] = max(cell_bbox[2], word_bbox[2])
                        cell_bbox[3] = max(cell_bbox[3], word_bbox[3])
        cell_bboxes[cell_num] = cell_bbox
    
    return cell_bboxes, inliers


def string_similarity(string1, string2):
    return SequenceMatcher(None, string1, string2).ratio()


# My current theory is that this is the correct code but that some examples are simply wrong
# (for example, the bolded text is aligned correctly but not the normal text)
def adjust_bbox_coordinates(data, doc):
    # Change bbox coordinates to be relative to PyMuPDF page.rect coordinate space
    media_box = doc[0].mediabox
    mat = doc[0].transformation_matrix

    for cell in data['html']['cells']:
        if not 'bbox' in cell:
            continue
        bbox = list(Rect(cell['bbox']) * mat)
        bbox = [bbox[0] + media_box[0],
                bbox[1] - media_box[1],
                bbox[2] + media_box[0],
                bbox[3] - media_box[1]]
        cell['bbox'] = bbox


def create_document_page_image(doc, page_num, zoom=None, output_image_max_dim=1000):
    page = doc[page_num]
    
    if zoom is None:
        zoom = output_image_max_dim / max(page.rect)
        
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix = mat, alpha = False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    return img


class AnnotationMismatchException(Exception):
    pass

class HTMLParseOverlappingGridCellsException(Exception):
    pass

class HTMLParseMissingGridCellsException(Exception):
    pass

class SmallTableException(Exception):
    pass

class AmbiguousHeaderException(Exception):
    pass

class OversizedHeaderException(Exception):
    pass

class RowColumnOverlapException(Exception):
    pass

# Average edit distance > 0.05
class TextAnnotationQualityException(Exception):
    pass

class UndeterminedRowBoundaryException(Exception):
    pass

class UndeterminedColumnBoundaryException(Exception):
    pass

# For cases where the data contains a cent symbol in its own column/cell, etc.
class OversegmentedColumnsException(Exception):
    pass

# For cases where the iterative cell text bounding box adjustment doesn't quickly converge
class RunawayTextAdjustmentException(Exception):
    pass

# For cases where the same grid cell is assigned to multiple spanning cells
class AmbiguousSpanningCellException(Exception):
    pass

class RowsIntersectException(Exception):
    pass

class ColumnsIntersectException(Exception):
    pass

class DotsInCellTextBboxException(Exception):
    pass

class PoorTextCellFitException(Exception):
    pass

# Use for interrupting after a specific event occurs for debugging
class DebugException(Exception):
    pass

# Specific to this dataset; cells at the top of the table can be incorrectly merged
# Merged cells at the top of the table are not inherently bad, but for this dataset we need to catch these
class OvermergedCellsException(Exception):
    pass

# Headers that are incomplete and stopped at a projected row header
class IncompleteHeaderException(Exception):
    pass

class NoTableBodyException(Exception):
    pass

class DotsRetainedException(Exception):
    pass

class BadProjectedRowHeaderException(Exception):
    pass

class MultipleColumnHeadersException(Exception):
    pass

class TextAnnotationQualityException(Exception):
    pass


def create_table_dict(annotation_data):
    table_dict = {}
    table_dict['reject'] = []
    table_dict['fix'] = []
    
    cells = []
    for cell in annotation_data['cells']:
        new_cell = {}
        new_cell['text_content'] = ' '.join(cell['content']).strip()
        new_cell['pdf_text_tight_bbox'] = []
        new_cell['column_nums'] = list(range(cell['start_col'], cell['end_col']+1))
        new_cell['row_nums'] = list(range(cell['start_row'], cell['end_row']+1))
        new_cell['is_column_header'] = False
        cells.append(new_cell)
        
    # Make sure no grid locations are duplicated
    # Could be bad data or bad parsing algorithm
    grid_cell_locations = []
    for cell in cells:
        for row_num in cell['row_nums']:
            for column_num in cell['column_nums']:
                grid_cell_locations.append((row_num, column_num))
    if not len(grid_cell_locations) == len(set(grid_cell_locations)):
        table_dict['reject'].append("HTML overlapping grid cells")
        
    num_rows = max([max(cell['row_nums']) for cell in cells]) + 1
    num_columns = max([max(cell['column_nums']) for cell in cells]) + 1
        
    table_dict['cells'] = cells
    table_dict['rows'] = {row_num: {'is_column_header': False} for row_num in range(num_rows)}
    table_dict['columns'] = {column_num: {} for column_num in range(num_columns)}
    
    return table_dict


def complete_table_grid(table_dict):
    rects_by_row = defaultdict(lambda: [None, None, None, None])
    rects_by_column = defaultdict(lambda: [None, None, None, None])
    table_rect = Rect()

    # Determine bounding box for rows and columns
    for cell in table_dict['cells']:
        if not 'pdf_text_tight_bbox' in cell or len(cell['pdf_text_tight_bbox']) == 0:
            continue

        bbox = cell['pdf_text_tight_bbox'] 

        table_rect.include_rect(list(bbox))
        
        min_row = min(cell['row_nums'])
        if rects_by_row[min_row][1] is None:
            rects_by_row[min_row][1] = bbox[1]
        else:
            rects_by_row[min_row][1] = min(rects_by_row[min_row][1], bbox[1])
            
        max_row = max(cell['row_nums'])
        if rects_by_row[max_row][3] is None:
            rects_by_row[max_row][3] = bbox[3]
        else:
            rects_by_row[max_row][3] = max(rects_by_row[max_row][3], bbox[3])
            
        min_column = min(cell['column_nums'])
        if rects_by_column[min_column][0] is None:
            rects_by_column[min_column][0] = bbox[0]
        else:
            rects_by_column[min_column][0] = min(rects_by_column[min_column][0], bbox[0])
            
        max_column = max(cell['column_nums'])
        if rects_by_column[max_column][2] is None:
            rects_by_column[max_column][2] = bbox[2]
        else:
            rects_by_column[max_column][2] = max(rects_by_column[max_column][2], bbox[2])

    table_bbox = list(table_rect)
    table_dict['pdf_table_bbox'] = table_bbox

    for row_num, row_rect in rects_by_row.items():
        row_rect[0] = table_bbox[0]
        row_rect[2] = table_bbox[2]

    for col_num, col_rect in rects_by_column.items():
        col_rect[1] = table_bbox[1]
        col_rect[3] = table_bbox[3]
        
    for k, row in table_dict['rows'].items():
        v = rects_by_row[k]
        table_dict['rows'][k]['pdf_row_bbox'] = list(v)
    for k, column in table_dict['columns'].items():
        v = rects_by_column[k]
        table_dict['columns'][k]['pdf_column_bbox'] = list(v)
        
    for k, row in table_dict['rows'].items():    
        for elem in row['pdf_row_bbox']:
            if elem is None:
                table_dict['reject'].append("undetermined row boundary")
    for k, column in table_dict['columns'].items():
        for elem in column['pdf_column_bbox']:
            if elem is None:
                table_dict['reject'].append("undetermined column boundary")
                
    # Adjust bounding boxes if minor overlap
    fixed_overlap = False
    num_rows = len(table_dict['rows'])
    for row_num in range(num_rows-1):
        row1_bbox = table_dict['rows'][row_num]['pdf_row_bbox']
        row2_bbox = table_dict['rows'][row_num+1]['pdf_row_bbox']
        overlap1 = overlap(row1_bbox, row2_bbox)
        overlap2 = overlap(row2_bbox, row1_bbox)
        
        if overlap1 > 0 and overlap2 > 0:
            if overlap1 < 0.18 and overlap2 < 0.18:
                fixed_overlap = True
                midpoint = 0.5 * (row1_bbox[3] + row2_bbox[1])
                table_dict['rows'][row_num]['pdf_row_bbox'][3] = midpoint
                table_dict['rows'][row_num+1]['pdf_row_bbox'][1] = midpoint
                fixed_overlap = True
            else:
                table_dict['reject'].append("rows intersect")
        
    # Intersect each row and column to determine grid cell bounding boxes
    #page_words = page.get_text_words()
    for cell in table_dict['cells']:
        rows_rect = Rect()
        cols_rect = Rect()

        for row_num in cell['row_nums']:
            rows_rect.include_rect(table_dict['rows'][row_num]['pdf_row_bbox'])

        for col_num in cell['column_nums']:
            cols_rect.include_rect(table_dict['columns'][col_num]['pdf_column_bbox'])

        pdf_bbox = rows_rect.intersect(cols_rect)
        cell['pdf_bbox'] = list(pdf_bbox)


def identify_projected_row_headers(table_dict):
    num_cols = len(table_dict['columns'])
    cells_with_text_count_by_row = defaultdict(int)
    all_cells_in_row_only_in_one_row_by_row = defaultdict(lambda: True)
    has_first_column_cell_with_text_by_row = defaultdict(bool)
    for cell in table_dict['cells']:
        if len(cell['text_content']) > 0:
            for row_num in cell['row_nums']:
                cells_with_text_count_by_row[row_num] += 1

            if 0 in cell['column_nums']:
                has_first_column_cell_with_text_by_row[row_num] = True

        one_row_only = len(cell['row_nums']) == 1
        for row_num in cell['row_nums']:
            all_cells_in_row_only_in_one_row_by_row[row_num] = all_cells_in_row_only_in_one_row_by_row[row_num] and one_row_only

    projected_row_header_rows = set()
    for row_num, row in table_dict['rows'].items():
        if (not row['is_column_header'] and cells_with_text_count_by_row[row_num] == 1
                and all_cells_in_row_only_in_one_row_by_row[row_num]
                and has_first_column_cell_with_text_by_row[row_num]):
            projected_row_header_rows.add(row_num)
            
    return projected_row_header_rows

def annotate_projected_row_headers(table_dict):    
    num_cols = len(table_dict['columns'])
    projected_row_header_rows = identify_projected_row_headers(table_dict)

    cells_to_remove = []
    for cell in table_dict['cells']:
        if len(set(cell['row_nums']).intersection(projected_row_header_rows)) > 0:
            if len(cell['text_content']) > 0:
                cell['column_nums'] = list(range(num_cols))
                cell['is_projected_row_header'] = True
            else:
                cells_to_remove.append(cell) # Consolidate blank cells after the first cell into the projected row header
        else:
            cell['is_projected_row_header'] = False

    for cell in cells_to_remove:
        table_dict['fix'].append('merged projected row header')
        table_dict['cells'].remove(cell)
        
    for row_num, row in table_dict['rows'].items():
        if row_num in projected_row_header_rows:
            row['is_projected_row_header'] = True
        else:
            row['is_projected_row_header'] = False
            
    # Delete projected row headers in last rows
    num_rows = len(table_dict['rows'])
    row_nums_to_delete = []
    for row_num in range(num_rows-1, -1, -1):
        if table_dict['rows'][row_num]['is_projected_row_header']:
            row_nums_to_delete.append(row_num)
        else:
            break
            
    if len(row_nums_to_delete) > 0:
        for row_num in row_nums_to_delete:
            del table_dict['rows'][row_num]
            table_dict['fix'].append('removed projected row header at bottom of table')
            for cell in table_dict['cells'][:]:
                if row_num in cell['row_nums']:
                    table_dict['cells'].remove(cell)


def merge_group(table_dict, group):
    cells_to_delete = []
    if len(group) == 1:
        return table_dict
    group = sorted(group, key=lambda k: min(k['row_nums'])) 
    cell = group[0]
    try:
        cell_text_rect = Rect(cell['pdf_text_tight_bbox'])
    except:
        cell_text_rect = Rect()
    for cell2 in group[1:]:
        cell['row_nums'] = list(set(sorted(cell['row_nums'] + cell2['row_nums'])))
        cell['column_nums'] = list(set(sorted(cell['column_nums'] + cell2['column_nums'])))
        cell['text_content'] = (cell['text_content'].strip() + " " + cell2['text_content'].strip()).strip()
        try:
            cell2_text_rect = Rect(cell2['pdf_text_tight_bbox'])
        except:
            cell2_text_rect = Rect()
        cell_text_rect = cell_text_rect.include_rect(list(cell2_text_rect))
        if cell_text_rect.get_area() == 0:
            cell['pdf_text_tight_bbox'] = []
        else:
            cell['pdf_text_tight_bbox'] = list(cell_text_rect)
        cell['is_projected_row_header'] = False
        cells_to_delete.append(cell2)
        
    try:
        for cell in cells_to_delete:
            table_dict['cells'].remove(cell)
            table_dict['fix'].append('merged oversegmented spanning cell')
    except:
        table_dict['reject'].append("ambiguous spanning cell")
        #raise AmbiguousSpanningCellException


def remove_empty_rows(table_dict):
    num_rows = len(table_dict['rows'])
    num_columns = len(table_dict['columns'])
    has_content_by_row = defaultdict(bool)
    for cell in table_dict['cells']:
        has_content = len(cell['text_content'].strip()) > 0
        for row_num in cell['row_nums']:
            has_content_by_row[row_num] = has_content_by_row[row_num] or has_content
    row_num_corrections = np.cumsum([int(not has_content_by_row[row_num]) for row_num in range(num_rows)]).tolist()
    
    # Delete cells in empty rows and renumber other cells
    cells_to_delete = []
    for cell in table_dict['cells']:
        new_row_nums = []
        for row_num in cell['row_nums']:
            if has_content_by_row[row_num]:
                new_row_nums.append(row_num - row_num_corrections[row_num])
        cell['row_nums'] = new_row_nums
        if len(new_row_nums) == 0:
            cells_to_delete.append(cell)
    for cell in cells_to_delete:
        table_dict['fix'].append('removed empty row')
        table_dict['cells'].remove(cell)
    
    rows = {}
    for row_num, has_content in has_content_by_row.items():
        if has_content:
            new_row_num = row_num - row_num_corrections[row_num]
            rows[new_row_num] = table_dict['rows'][row_num]
    table_dict['rows'] = rows
    
def merge_rows(table_dict):
    num_rows = len(table_dict['rows'])
    num_columns = len(table_dict['columns'])
    co_occurrence_matrix = np.zeros((num_rows, num_rows))
    for cell in table_dict['cells']:
        for row_num1 in cell['row_nums']:
            for row_num2 in cell['row_nums']:
                if row_num1 >= row_num2:
                    continue
                co_occurrence_matrix[row_num1, row_num2] += len(cell['column_nums'])
                
    new_row_num = 0
    current_row_group = 0
    keep_row = [True]
    row_grouping = [current_row_group]
    for row_num in range(num_rows-1):
        if not co_occurrence_matrix[row_num, row_num+1] == num_columns:
            keep_row.append(True)
            new_row_num += 1
        else:
            table_dict['fix'].append('merged rows spanned together in every column')
            keep_row.append(False)
        row_grouping.append(new_row_num)

    for cell in table_dict['cells']:
        cell['row_nums'] = [row_grouping[row_num] for row_num in cell['row_nums'] if keep_row[row_num]]
        
    table_dict['rows'] = {row_grouping[row_num]: table_dict['rows'][row_num] for row_num in range(num_rows) if keep_row[row_num]} 
            
        
def remove_empty_columns(table_dict):
    num_rows = len(table_dict['rows'])
    num_columns = len(table_dict['columns'])
    has_content_by_column = defaultdict(bool)
    for cell in table_dict['cells']:
        has_content = len(cell['text_content'].strip()) > 0
        for column_num in cell['column_nums']:
            has_content_by_column[column_num] = has_content_by_column[column_num] or has_content
    column_num_corrections = np.cumsum([int(not has_content_by_column[column_num]) for column_num in range(num_columns)]).tolist()
    
    # Delete cells in empty columns and renumber other cells
    cells_to_delete = []
    for cell in table_dict['cells']:
        new_column_nums = []
        for column_num in cell['column_nums']:
            if has_content_by_column[column_num]:
                new_column_nums.append(column_num - column_num_corrections[column_num])
        cell['column_nums'] = new_column_nums
        if len(new_column_nums) == 0:
            cells_to_delete.append(cell)
    for cell in cells_to_delete:
        table_dict['fix'].append('removed empty column')
        table_dict['cells'].remove(cell)
    
    columns = {}
    for column_num, has_content in has_content_by_column.items():
        if has_content:
            new_column_num = column_num - column_num_corrections[column_num]
            columns[new_column_num] = table_dict['columns'][column_num]
    table_dict['columns'] = columns
    
def merge_columns(table_dict):
    num_rows = len(table_dict['rows'])
    num_columns = len(table_dict['columns'])
    co_occurrence_matrix = np.zeros((num_columns, num_columns))
    for cell in table_dict['cells']:
        for column_num1 in cell['column_nums']:
            for column_num2 in cell['column_nums']:
                if column_num1 >= column_num2:
                    continue
                co_occurrence_matrix[column_num1, column_num2] += len(cell['row_nums'])
                
    new_column_num = 0
    current_column_group = 0
    keep_column = [True]
    column_grouping = [current_column_group]
    for column_num in range(num_columns-1):
        if not co_occurrence_matrix[column_num, column_num+1] == num_rows:
            keep_column.append(True)
            new_column_num += 1
        else:
            table_dict['fix'].append('merged columns spanned together in every row')
            keep_column.append(False)
        column_grouping.append(new_column_num)

    for cell in table_dict['cells']:
        cell['column_nums'] = [column_grouping[column_num] for column_num in cell['column_nums'] if keep_column[column_num]]
        
    table_dict['columns'] = {column_grouping[column_num]: table_dict['columns'][column_num] for column_num in range(num_columns) if keep_column[column_num]}


# Look for tables with blank cells to merge in the first column
def merge_spanning_cells_in_first_column(table_dict):
    first_column_cells = [cell for cell in table_dict['cells'] if 0 in cell['column_nums']]
    first_column_cells = sorted(first_column_cells, key=lambda item: max(item['row_nums']))
    
    first_column_merge_exclude = set()
    
    # Look for blank cells at bottom of first column
    text_by_row_num = {}
    for cell in table_dict['cells']:
        if 0 in cell['column_nums']:
            for row_num in cell['row_nums']:
                if not cell['is_column_header']:
                    text_by_row_num[row_num] = cell['text_content'].strip()
                else:
                    text_by_row_num[row_num] = "_"
    bottom_blank_rows = set()
    blank_rows = set()
    still_bottom = True
    add_bottom_rows = True
    for row_num in sorted(table_dict['rows'].keys(), reverse=True):
        if len(text_by_row_num[row_num]) > 0:
            still_bottom = False
        elif still_bottom:
            bottom_blank_rows.add(row_num)
        else:
            add_bottom_rows = False
            break
    if add_bottom_rows:
        first_column_merge_exclude = first_column_merge_exclude.union(bottom_blank_rows)
    
    
    # Look for tables with multiple headers
    num_rows = len(table_dict['rows'])
    num_columns = len(table_dict['columns'])
    cell_grid = np.zeros((num_rows, num_columns)).astype('str').tolist()
    for cell in table_dict['cells']:
        for row_num in cell['row_nums']:
            for column_num in cell['column_nums']:
                cell_grid[row_num][column_num] = cell['text_content']
    for row_num1 in range(num_rows-1):
        row1 = table_dict['rows'][row_num1]
        if not row1['is_column_header']:
            continue
        for row_num2 in range(row_num1+1, num_rows):
            row2 = table_dict['rows'][row_num2]
            if row2['is_column_header']:
                continue
            if cell_grid[row_num1] == cell_grid[row_num2]:
                first_column_merge_exclude.add(row_num2)
    for cell1 in table_dict['cells']:
        for cell2 in table_dict['cells']:
            if cell1['is_column_header'] and not cell2['is_column_header']:
                if cell1['text_content'] == cell2['text_content'] and len(cell1['text_content'].strip()) > 0:
                    for row_num in cell2['row_nums']:
                        first_column_merge_exclude.add(row_num)
    
    current_filled_cell = None
    groups = defaultdict(list)
    group_num = -1
    for cell in first_column_cells:
        if len(set(cell['row_nums']).intersection(first_column_merge_exclude)) > 0:
            group_num += 1
        elif len(cell['text_content']) > 0:
            group_num += 1
        if group_num >= 0:
            groups[group_num].append(cell)
        
    for group_num, group in groups.items():
        if len(group) > 1 and not group[0]['is_projected_row_header'] and not group[0]['is_column_header']:
            merge_group(table_dict, group)


# STANDARDS:
# 1. Column header, if it exists, is a tree structure. FinTabNet contains no header annotation so we can only
#    infer the header given some assumptions. If the top row does not contain all leaf nodes, complete the tree down to the
#    leaf nodes.
# 2. There should be no blank cells in the column header. Blank cells should be aggregated into supercells where possible.
#    - First, blank supercells should be split into blank grid cells.
#    - If a column header cell has only blank grid cells directly below it, extend the cell downward to consume
#      any rows of entirely blank cells directly below it.
#    - After doing this for all column header cells, if a column header cell has only blank cells above it, consume 
#      any rows of entirely blank cells directly above it.
#    - Blank supercells that occur after this grouping are arguably rightly annotated as supercells, but we will not
#      annotate these as supercells at the moment for consistency with previous datasets that annotated these as blank
#      cells only. Detecting a blank supercell would not impact the structure inferred for the table.
#    - Any remaining blank cells are ambiguous, and while it is not good table design to have these, they're not likely
#      to be a nuisance.
# 3. There should be no blank cells in the row header. This is trickier because the row header is not explicitly
#    annotated and must be inferred. See below for more on determining which columns are in the row header.
#    - For columns in the row header, blank cells should be aggregated under the first cell that does not span the
#      entire row. This assumes "top" vertical alignment for text. "Middle" vertical alignment is normally already
#      associated with supercells and is already explicit.
# 4. Inferring the row header.
#    - The row header is explicit whenever the first N columns do not have a column header. In other words, when
#      the stub header is blank. Otherwise it is implicit which columns, if any, correspond to the column header.
#    - If a row header exists, it is also a tree just like the column header and must end at a column of leaf nodes.
#      Not only does this mean supercells cannot be the final column of a row header, but repeated values in a column
#      mean that the column cannot be the final column of a row header.
#    - A column that is not part of the row header (possibly the first column) can have repeated values. Having repeated
#      values is an indication of row header continuation but not of row header status to begin with.
#    - In most cases, numeric values are data. If the numeric values are integer and sorted, this may be part of the row
#      header.
#    - Rows where only one cell has content, either left justified or centered across the table, are part of an implicit
#      first column that begins a row header. The stub header belongs in this first column if there is not a row cell.
# 5. A row cell at the top of the table is either the title of the table (if there are no other row cells in the table),
#    or part of the row header if there are additional row cells below, and belongs in an implicit column.
# 6. Tables have at least one row and two columns. A table with only one column is a list.

def correct_header(table_dict, assume_header_if_more_than_two_columns=True):
    num_columns = len(table_dict['columns'])
    num_rows = len(table_dict['rows'])
    
    if num_columns < 2 or num_rows < 1:
        table_dict['reject'].append("small table")
        #raise SmallTableException("Table does not have at least one row and two columns") 
        
    #---DETERMINE FULL EXTENT OF COLUMN HEADER
    # - Each of the below steps determines different rows that must be in the column header.
    # - The final column header includes all rows that are originally annotated as being in the column
    #   header plus any additional rows determined to be in the column header by the following steps.
    
    table_has_column_header = False
    
    # First determine if there is definitely a column header. Four cases:
    
    # 1. We specify that we want to assume there is one for all tables with more than two columns:
    if assume_header_if_more_than_two_columns and num_columns > 2:
        table_has_column_header = True
    
    # 2. An annotator says there is
    if not table_has_column_header:
        header_rows = [row_num for row_num, row in table_dict['rows'].items() if row['is_column_header']]
        if 0 in header_rows:
            table_has_column_header = True
        
    # 3. The cell occupying the first row and column is blank
    if not table_has_column_header:
        for cell in table_dict['cells']:
            if 0 in cell['column_nums'] and 0 in cell['row_nums'] and len(cell['text_content'].strip()) == 0:
                table_has_column_header = True
                break
    
    # 4. There is a horizontal spanning cell in the first row
    if not table_has_column_header:
        for cell in table_dict['cells']:
            if 0 in cell['row_nums'] and len(cell['column_nums']) > 1:
                table_has_column_header = True
                break

    # Then determine if the column header needs to be extended past its current annotated extent.
    #  1. A header that already is annotated in at least one row continues at least until each column
    #     has a cell occupying only that column
    #  2. A header with a column with a blank cell must continue at least as long as the blank cells continue
    #     (unless rule #1 is satisfied and a possible projected row header is reached?)
    if table_has_column_header:
        # Do not use this rule; while perhaps not ideal, columns can have the same header
        #print("Flattening header")
        #num_rows = len(table_dict['rows'])
        #num_columns = len(table_dict['columns'])
        #cell_grid = np.zeros((num_rows, num_columns)).astype('str').tolist()
        #for cell in table_dict['cells']:
        #    for row_num in cell['row_nums']:
        #        for column_num in cell['column_nums']:
        #            cell_grid[row_num][column_num] = cell['text_content']
        #flattened_header = ['' for column_num in range(num_columns)]
        #for row_num in range(num_rows):
        #    unique_headers = True
        #    for column_num in range(num_columns):
        #        flattened_header[column_num] += ' ' + cell_grid[row_num][column_num]
        #        flattened_header[column_num] = flattened_header[column_num].strip()
        #    print(flattened_header)
        #    for column_num1 in range(num_columns-1):
        #        for column_num2 in range(column_num1+1, num_columns):
        #            if flattened_header[column_num1] == flattened_header[column_num2] and len(flattened_header[column_num1]) > 0:
        #                unique_headers = False
        #    if unique_headers:
        #        break
        #unique_header_row = row_num
        #print(unique_header_row)
        
        first_column_filled_by_row = defaultdict(bool)
        for cell in table_dict['cells']:
            if 0 in cell['column_nums']:
                if len(cell['text_content']) > 0:
                    for row_num in cell['row_nums']:
                        first_column_filled_by_row[row_num] = True        
        
        first_column_filled_by_row = defaultdict(bool)
        for cell in table_dict['cells']:
            if 0 in cell['column_nums']:
                if len(cell['text_content']) > 0:
                    for row_num in cell['row_nums']:
                        first_column_filled_by_row[row_num] = True
        
        first_single_node_row_by_column = defaultdict(lambda: len(table_dict['rows'])-1)
        for cell in table_dict['cells']:
            if len(cell['column_nums']) == 1:
                first_single_node_row_by_column[cell['column_nums'][0]] = min(first_single_node_row_by_column[cell['column_nums'][0]],
                                                                               max(cell['row_nums']))
                
        first_filled_single_node_row_by_column = defaultdict(lambda: len(table_dict['rows'])-1)
        for cell in table_dict['cells']:
            if len(cell['column_nums']) == 1 and len(cell['text_content'].strip()) > 0:
                first_filled_single_node_row_by_column[cell['column_nums'][0]] = min(first_filled_single_node_row_by_column[cell['column_nums'][0]],
                                                                               max(cell['row_nums']))
                
        first_filled_cell_by_column = defaultdict(lambda: len(table_dict['rows'])-1)
        for cell in table_dict['cells']:
            if len(cell['text_content']) > 0:
                min_row_num = min(cell['row_nums'])
                for column_num in cell['column_nums']:
                    first_filled_cell_by_column[column_num] = min(first_filled_cell_by_column[column_num],
                                                                  min_row_num)
                    
        projected_row_header_rows = identify_projected_row_headers(table_dict)
        if 0 in projected_row_header_rows:
            table_dict['reject'].append("bad projected row header")
            #raise BadProjectedRowHeaderException('Starting with PRH')
        #for row_num in range(num_rows):
        #    if row_num in projected_row_header_rows:
        #        projected_row_header_rows.remove(row_num)
        #    else:
        #        break
        
        # Header must continue until at least this row
        minimum_grid_cell_single_node_row = max(first_single_node_row_by_column.values())
        
        # Header can stop prior to the first of these rows that occurs after the above row
        minimum_first_body_row = min(num_rows-1, max(first_filled_cell_by_column.values()))
        
        # Determine the max row for which a column N has been single and filled but column N+1 has not
        minimum_all_following_filled = -1
        for row_num in range(num_rows):
            for column_num1 in range(num_columns-1):
                for column_num2 in range(column_num1+1, num_columns):
                    if (first_filled_single_node_row_by_column[column_num2] > row_num
                        and first_filled_single_node_row_by_column[column_num1] < first_filled_single_node_row_by_column[column_num2]):
                        minimum_all_following_filled = row_num + 1

        #minimum_projected_row_header_row = min([num_rows-1] + [elem for elem in projected_row_header_rows if elem > minimum_grid_cell_single_node_row])
        if len(projected_row_header_rows) > 0:
            minimum_projected_row_header_row = min(projected_row_header_rows)
        else:
            minimum_projected_row_header_row = num_rows

        #first_possible_last_header_row = min(minimum_first_body_row, minimum_projected_row_header_row) - 1
        first_possible_last_header_row = minimum_first_body_row - 1
                    
        last_header_row = max(minimum_all_following_filled,
                              minimum_grid_cell_single_node_row,
                              first_possible_last_header_row)
        
        x = last_header_row
        while(last_header_row < num_rows and not first_column_filled_by_row[last_header_row+1]):
            last_header_row += 1            
        
        #incomplete_header = False # temp for debugging
        if minimum_projected_row_header_row <= last_header_row:
            last_header_row = minimum_projected_row_header_row - 1
            #incomplete_header = True
        
        for cell in table_dict['cells']:
            if max(cell['row_nums']) <= last_header_row:
                cell['is_column_header'] = True
        
        for row_num, row in table_dict['rows'].items():
            if row_num <= last_header_row:
                row['is_column_header'] = True
                
        #if not x == last_header_row:
        #    raise DebugException("Header extended")
                
        #if minimum_all_following_filled == last_header_row:
        #    raise DebugException
        
        #if incomplete_header:
        #    raise IncompleteHeaderException("Set last header row to be just before minimum projected row header row".format(last_header_row, minimum_projected_row_header_row))
    
    if not table_has_column_header and num_columns == 2:
        table_dict['reject'].append("ambiguous header")
        #raise AmbiguousHeaderException("Missing header annotation for table with two columns; cannot unambiguously determine header")

def canonicalize(table_dict):
    # Preprocessing step: Split every blank spanning cell in the column header into blank grid cells.
    cells_to_delete = []
    try:
        for cell in table_dict['cells']:
            if (cell['is_column_header'] and len(cell['text_content'].strip()) == 0
                    and (len(cell['column_nums']) > 1 or len(cell['row_nums']) > 1)):
                cells_to_delete.append(cell)
                # Split this blank spanning cell into blank grid cells
                for column_num in cell['column_nums']:
                    for row_num in cell['row_nums']:
                        #row_bbox = table_dict['rows'][row_num]['pdf_row_bbox']
                        #column_bbox = table_dict['columns'][column_num]['pdf_column_bbox']
                        #bbox = list(Rect(row_bbox).intersect(list(column_bbox)))
                        new_cell = {'text_content': '',
                                    'column_nums': [column_num],
                                    'row_nums': [row_num],
                                    'is_column_header': cell['is_column_header'],
                                    'pdf_text_tight_bbox': [],
                                    'is_projected_row_header': False}
                        table_dict['cells'].append(new_cell)
    except:
        print(traceback.format_exc())
    for cell in cells_to_delete:
        table_dict['cells'].remove(cell)
        
    # Index cells by row-column position
    cell_grid_index = {}
    for cell in table_dict['cells']:
        for column_num in cell['column_nums']:
            for row_num in cell['row_nums']:
                cell_grid_index[(row_num, column_num)] = cell
        
    # Go bottom up, try to extend non-blank cells up to absorb blank cells
    header_groups = []
    for cell in table_dict['cells']:
        if not cell['is_column_header'] or len(cell['text_content']) == 0:
            continue
        header_group = [cell]
        next_row_num = min(cell['row_nums']) - 1
        for row_num in range(next_row_num, -1, -1):
            all_are_blank = True
            for column_num in cell['column_nums']:
                cell2 = cell_grid_index[(row_num, column_num)]
                all_are_blank = all_are_blank and len(cell2['text_content']) == 0
            if all_are_blank:
                for column_num in cell['column_nums']:
                    header_group.append(cell_grid_index[(row_num, column_num)])
            else:
                break # Stop looking; must be contiguous
        if len(header_group) > 1:
            header_groups.append(header_group)
    for group in header_groups:
        merge_group(table_dict, group)
            
    # Index cells by row-column position
    cell_grid_index = {}
    for cell in table_dict['cells']:
        for column_num in cell['column_nums']:
            for row_num in cell['row_nums']:
                cell_grid_index[(row_num, column_num)] = cell
                
    num_rows = len(table_dict['rows'])
    # Go top down, try to extend non-blank cells down to absorb blank cells
    header_groups = []
    for cell in table_dict['cells']:
        if not cell['is_column_header'] or len(cell['text_content']) == 0:
            continue
        header_group = [cell]
        next_row_num = max(cell['row_nums']) + 1
        for row_num in range(next_row_num, num_rows):
            if not table_dict['rows'][row_num]['is_column_header']:
                break
            all_are_blank = True
            for column_num in cell['column_nums']:
                cell2 = cell_grid_index[(row_num, column_num)]
                all_are_blank = all_are_blank and len(cell2['text_content']) == 0
            if all_are_blank:
                for column_num in cell['column_nums']:
                    header_group.append(cell_grid_index[(row_num, column_num)])
            else:
                break # Stop looking; must be contiguous
        if len(header_group) > 1:
            header_groups.append(header_group)
    for group in header_groups:
        merge_group(table_dict, group)
    
    # Index cells by row-column position
    cell_grid_index = {}
    for cell in table_dict['cells']:
        for column_num in cell['column_nums']:
            for row_num in cell['row_nums']:
                cell_grid_index[(row_num, column_num)] = cell
        
    # Go top down, merge any neighboring cells occupying the same columns, whether they are blank or not
    header_groups_by_row_column = defaultdict(list)
    header_groups = []
    do_full_break = False
    for row_num in table_dict['rows']:
        for column_num in table_dict['columns']:
            cell = cell_grid_index[(row_num, column_num)]
            if not cell['is_column_header']:
                do_full_break = True
                break
            if len(header_groups_by_row_column[(row_num, column_num)]) > 0:
                continue
            if not row_num == min(cell['row_nums']) and column_num == min(cell['column_nums']):
                continue
            # Start new header group
            header_group = [cell]
            next_row_num = max(cell['row_nums']) + 1
            while next_row_num < num_rows:
                cell2 = cell_grid_index[(next_row_num, column_num)]
                if cell2['is_column_header'] and set(cell['column_nums']) == set(cell2['column_nums']):
                    header_group.append(cell2)
                    for row_num2 in cell2['row_nums']:
                        for column_num2 in cell2['column_nums']:
                            header_groups_by_row_column[(row_num2, column_num2)] = header_group
                else:
                    break
                next_row_num = max(cell2['row_nums']) + 1
            for row_num2 in cell['row_nums']:
                for column_num2 in cell['column_nums']:
                    header_groups_by_row_column[(row_num2, column_num2)] = header_group
            if len(header_group) > 1:
                header_groups.append(header_group)
        if do_full_break:
            break
    for group in header_groups:
        merge_group(table_dict, group)
        
    # Merge spanning cells in the row header
    merge_spanning_cells_in_first_column(table_dict)


def is_all_dots(text):
    if len(text) > 0 and len(text.replace('.','')) == 0:
        return True
    return False

def extract_pdf_text(table_dict, page_words, threshold=0.5):
    adjusted_text_tight_bbox = False
    for cell in table_dict['cells']:
        pdf_text_tight_bbox = cell['pdf_text_tight_bbox']
        pdf_bbox = cell['pdf_bbox']
        
        cell_page_words = [w for w in page_words if Rect(w[:4]).intersect(list(pdf_bbox)).get_area() / Rect(w[:4]).get_area() > threshold]
        cell_words = [w[4] for w in cell_page_words]
        cell_text = ''.join(cell_words)
        
        # Remove trailing dots from cell_page_words
        # Some of the original annotations include dots in the pdf_text_tight_bbox when they shouldn't
        # This code ensures that those are fixed, plus that dots are not added by extracting text from the
        # entire grid cell
        if len(cell_text) > 2 and cell_text[-1] == '.' and cell_text[-2] == '.':
            for page_word in cell_page_words[::-1]:
                if is_all_dots(page_word[4]):
                    table_dict['fix'].append('removed dots from text cell')
                    cell_page_words.remove(page_word)
                else:
                    break
        
        cell_words_rect = Rect()
        for w in cell_page_words:
            cell_words_rect.include_rect(w[:4])
        cell_words = [w[4] for w in cell_page_words]
        cell_text = ' '.join(cell_words)
        cell_text = cell_text.replace(' .', '.').replace(' ,', ',')
        if cell_text.endswith('..'):
            table_dict['reject'].append("dots retained")
            #raise DotsRetainedException("Dots retained in text [{}] '{}'".format(cell_words, cell_text))
        cell['pdf_text_content'] = cell_text
        if cell_words_rect.get_area() > 0:
            new_pdf_text_tight_bbox = list(cell_words_rect)
            if not pdf_text_tight_bbox == new_pdf_text_tight_bbox:
                adjusted_text_tight_bbox = True
                cell['pdf_text_tight_bbox'] = new_pdf_text_tight_bbox
                
    return adjusted_text_tight_bbox


def overlap(bbox1, bbox2):
    try:
        return Rect(bbox1).intersect(list(bbox2)).get_area() / Rect(bbox1).get_area()
    except:
        return 1

def table_text_edit_distance(cells):
    if len(cells) == 0:
        return 0
    
    D = 0
    for cell in cells:
        # Remove spaces and trailing periods
        xml_text = ''.join(cell['text_content'].split()).strip('.')
        pdf_text = ''.join(cell['pdf_text_content'].split()).strip('.')
        L = max(len(xml_text), len(pdf_text))
        if L > 0:
            D += editdistance.eval(xml_text, pdf_text) / L
            
    return D / len(cells)

def quality_control(table_dict, page_words):
    for row_num1, row1 in table_dict['rows'].items():
        for row_num2, row2, in table_dict['rows'].items():
            if row_num1 == row_num2 - 1:
                if row1['pdf_row_bbox'][3] > row2['pdf_row_bbox'][1] + 1:
                    table_dict['reject'].append("rows intersect")
                    #raise RowsIntersectException
                    
    for column_num1, column1 in table_dict['columns'].items():
        for column_num2, column2, in table_dict['columns'].items():
            if column_num1 == column_num2 - 1:
                if column1['pdf_column_bbox'][2] > column2['pdf_column_bbox'][0] + 1:
                    table_dict['reject'].append("columns intersect")
                    #raise ColumnsIntersectException
    
    D = table_text_edit_distance(table_dict['cells'])
    if D > 0.05:
        table_dict['reject'].append("text annotation quality")
        
    word_overlaps = []
    table_bbox = table_dict['pdf_table_bbox']
    for w in page_words:
        if w[4] == '.':
            continue
        if overlap(w[:4], table_bbox) < 0.5:
            continue
        word_overlaps.append(max([overlap(w[:4], cell['pdf_bbox']) for cell in table_dict['cells']]))
    C = sum(word_overlaps) / len(word_overlaps)
    if C < 0.9:
        table_dict['reject'].append("poor text cell fit")


def remove_html_tags_in_text(table_dict):
    for cell in table_dict['cells']:
        cell['text_content'] = cell['text_content'].replace("<i>", " ")
        cell['text_content'] = cell['text_content'].replace("</i>", " ")
        cell['text_content'] = cell['text_content'].replace("<sup>", " ")
        cell['text_content'] = cell['text_content'].replace("</sup>", " ")
        cell['text_content'] = cell['text_content'].replace("<sub>", " ")
        cell['text_content'] = cell['text_content'].replace("</sub>", " ")
        cell['text_content'] = cell['text_content'].replace("  ", " ")
        cell['text_content'] = cell['text_content'].strip()


def is_good_bbox(bbox, page_bbox):
    if (not bbox[0] is None and not bbox[1] is None and not bbox[2] is None and not bbox[3] is None
            and bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] <= page_bbox[2] and bbox[3] <= page_bbox[3]
            and bbox[0] < bbox[2]-1 and bbox[1] < bbox[3]-1):
        return True
    return False


def create_document_page_image(doc, page_num, output_image_max_dim=1000):
    page = doc[page_num]
    page_width = page.rect[2]
    page_height = page.rect[3]
    
    if page_height > page_width:
        zoom = output_image_max_dim / page_height
        output_image_height = output_image_max_dim
        output_image_width = int(round(output_image_max_dim * page_width / page_height))
    else:
        zoom = output_image_max_dim / page_width
        output_image_width = output_image_max_dim
        output_image_height = int(round(output_image_max_dim * page_height / page_width))
        
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix = mat, alpha = False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    return img


def create_pascal_voc_page_element(image_filename, output_image_width, output_image_height, database):
    # Create XML of tables on PDF page in PASCAL VOC format
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder").text = ""
    filename = ET.SubElement(annotation, "filename").text = image_filename
    path = ET.SubElement(annotation, "path").text = image_filename
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database").text = database
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width").text = str(output_image_width)
    height = ET.SubElement(size, "height").text = str(output_image_height)
    depth = ET.SubElement(size, "depth").text = "3"
    segmented = ET.SubElement(annotation, "segmented").text = "0"
    
    return annotation


def create_pascal_voc_object_element(class_name, bbox, page_bbox, output_image_max_dim=1000):
    bbox_area = fitz.Rect(bbox).get_area()
    if bbox_area == 0:
        raise Exception
    intersect_area = fitz.Rect(page_bbox).intersect(fitz.Rect(bbox)).get_area()
    if abs(intersect_area - bbox_area) > 0.1:
        print(bbox)
        print(bbox_area)
        print(page_bbox)
        print(intersect_area)
        raise Exception
    
    object_ = ET.Element("object")
    name = ET.SubElement(object_, "name").text = class_name
    pose = ET.SubElement(object_, "pose").text = "Frontal"
    truncated = ET.SubElement(object_, "truncated").text = "0"
    difficult = ET.SubElement(object_, "difficult").text = "0"
    occluded = ET.SubElement(object_, "occluded").text = "0"
    bndbox = ET.SubElement(object_, "bndbox")
    
    page_width = page_bbox[2] - page_bbox[0]
    page_height = page_bbox[3] - page_bbox[1]
    
    if page_width > page_height:
        output_image_width = output_image_max_dim
        output_image_height = int(output_image_max_dim * page_height / page_width)
    else:
        output_image_height = output_image_max_dim
        output_image_width = int(output_image_max_dim * page_width / page_height)

    xmin = (bbox[0] - page_bbox[0]) * output_image_width / page_width
    ymin = (bbox[1] - page_bbox[1]) * output_image_height / page_height
    xmax = (bbox[2] - page_bbox[0]) * output_image_width / page_width
    ymax = (bbox[3] - page_bbox[1]) * output_image_height / page_height
    
    ET.SubElement(bndbox, "xmin").text = str(xmin)
    ET.SubElement(bndbox, "ymin").text = str(ymin)
    ET.SubElement(bndbox, "xmax").text = str(xmax)
    ET.SubElement(bndbox, "ymax").text = str(ymax)
    
    return object_


def save_xml_pascal_voc(page_annotation, filepath):
    xmlstr = minidom.parseString(ET.tostring(page_annotation)).toprettyxml(indent="   ")
    with open(filepath, "w") as f:
        f.write(xmlstr)
        
        
def bbox_pdf_to_image(bbox, page_bbox, output_image_max_dim=1000):
    page_width = page_bbox[2] - page_bbox[0]
    page_height = page_bbox[3] - page_bbox[1]
    
    if page_width > page_height:
        output_image_width = output_image_max_dim
        output_image_height = int(output_image_max_dim * page_height / page_width)
    else:
        output_image_height = output_image_max_dim
        output_image_width = int(output_image_max_dim * page_width / page_height)

    xmin = (bbox[0] - page_bbox[0]) * output_image_width / page_width
    ymin = (bbox[1] - page_bbox[1]) * output_image_height / page_height
    xmax = (bbox[2] - page_bbox[0]) * output_image_width / page_width
    ymax = (bbox[3] - page_bbox[1]) * output_image_height / page_height
    
    return [xmin, ymin, xmax, ymax]


def get_tokens_in_table_img(page_words, table_img_bbox):
    tokens = []
    for word_num, word in enumerate(page_words):
        word['flags'] = 0
        word['span_num'] = word_num
        word['line_num'] = 0
        word['block_num'] = 0
        tokens.append(word)

    tokens_in_table = [token for token in tokens if utils.iob(token['bbox'], table_img_bbox) >= 0.5]
    
    return tokens_in_table


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        help="Root directory for source data to process")
    parser.add_argument('--output_dir',
                        help="Root directory for output data")
    parser.add_argument('--train_padding', type=int, default=30,
                        help="The amount of padding to add around a table in the training set when cropping.")
    parser.add_argument('--test_padding', type=int, default=5,
                        help="The amount of padding to add around a table in the val and test sets when cropping.")
    parser.add_argument('--skip_large', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()

    data_directory = args.data_dir

    output_json_directory = os.path.join(args.output_dir, "SciTSR.c-PDF_Annotations_JSON")
    if not os.path.exists(output_json_directory):
        os.makedirs(output_json_directory)

    output_subdirs = ['images', 'train', 'test', 'val']

    output_detection_directory = os.path.join(args.output_dir, "SciTSR.c-Image_Detection_PASCAL_VOC")
    if not os.path.exists(output_detection_directory):
        os.makedirs(output_detection_directory)
    for subdir in output_subdirs:
        subdirectory = os.path.join(output_detection_directory, subdir)
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)
        
    output_page_words_directory = os.path.join(args.output_dir, "SciTSR.c-Image_Page_Words_JSON")
    if not os.path.exists(output_page_words_directory):
        os.makedirs(output_page_words_directory)
    
    output_structure_directory = os.path.join(args.output_dir, "SciTSR.c-Image_Structure_PASCAL_VOC")
    if not os.path.exists(output_structure_directory):
        os.makedirs(output_structure_directory)
    for subdir in output_subdirs:
        subdirectory = os.path.join(output_structure_directory, subdir)
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)
        
    output_table_words_directory = os.path.join(args.output_dir, "SciTSR.c-Image_Table_Words_JSON")
    if not os.path.exists(output_table_words_directory):
        os.makedirs(output_table_words_directory)

    train_structure_files = os.listdir(os.path.join(data_directory, "train", "structure"))
    test_structure_files = os.listdir(os.path.join(data_directory, "test", "structure"))

    structure_filepaths = [os.path.join(data_directory, "train", "structure", elem) for elem in train_structure_files]
    structure_filepaths += [os.path.join(data_directory, "test", "structure", elem) for elem in test_structure_files]

    with open(os.path.join(data_directory, "train", "structure", structure_filepaths[1]), 'r') as infile:
        data = json.load(infile)

    splits_by_filepath = dict()

    test_filepaths = [os.path.join(data_directory, "test", "structure", elem) for elem in test_structure_files]
    train_filepaths = [os.path.join(data_directory, "train", "structure", elem) for elem in train_structure_files]

    for filepath in test_filepaths:
        splits_by_filepath[filepath] = 'test'

    n = len(train_filepaths)
    print(n)
    order = np.random.permutation(n)
    split_point = int(n * 0.875)
    for idx in order[:split_point]:
        splits_by_filepath[train_filepaths[idx]] = 'train'
    for idx in order[split_point:]:
        splits_by_filepath[train_filepaths[idx]] = 'val'

    processed_count = 0
    good_count = 0
    reject_count = 0
    reject_reasons = defaultdict(list)
    fixes = defaultdict(list)
    kept_as_is_count = 0

    output_image_max_dim = 1000

    do_break = False

    for idx, structure_filepath in tqdm(enumerate(structure_filepaths)):
        split = splits_by_filepath[structure_filepath]

        try:
            with open(os.path.join(data_directory, "train", "structure", structure_filepath), 'r') as infile:
                data = json.load(infile)
            data['cells'] = sorted(sorted(data['cells'], key=lambda x: x['start_col']), key=lambda x: x['start_row'])
            table_dict = create_table_dict(data)

            img_filepath = structure_filepath.replace("structure", "img").replace(".json", ".png")

            pdf_filepath = structure_filepath.replace("structure", "pdf").replace(".json", ".pdf")

            doc = fitz.open(pdf_filepath)
            page = doc[0]

            page_words = doc[0].get_text_words()

            if split == 'val' or split == 'test':
                padding = args.test_padding
            else:
                padding = args.train_padding

            # For SciTSR, the table isn't always completely inside the PDF page
            page_rect = page.mediabox
            for word in page_words:
                bbox = word[:4]
                bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[2]+padding]
                page_rect.include_rect(bbox)
            page.set_mediabox(page_rect)

            img = create_document_page_image(doc, 0, output_image_max_dim=1000)           

            cell_bboxes, inliers = locate_table(page_words, table_dict)

            for cell, cell_bbox in zip(table_dict['cells'], cell_bboxes.values()):
                if cell_bbox is None:
                    cell_bbox = []
                cell['pdf_text_tight_bbox'] = cell_bbox

            #adjust_bbox_coordinates(table_dict, doc)

            tables = [table_dict]
        except:
            traceback.print_exc()
            continue
        
        document_tables = []
        for table_index, table_dict in enumerate(tables):
            try:
                page_num = 0
                
                page = doc[page_num]

                exclude_for_structure = False
                exclude_for_detection = False

                table_dict['exclude_for_structure'] = exclude_for_structure
                table_dict['exclude_for_detection'] = exclude_for_detection
                
                table_dict['split'] = split
                table_dict['pdf_file_name'] = pdf_filepath.split("/")[-1]
                table_dict['pdf_page_index'] = page_num
                table_dict['document_id'] = table_dict['pdf_file_name'].replace(".pdf", "")
                table_dict['source_file_name'] = structure_filepath.split("/")[-1]
                table_dict['pdf_full_page_bbox'] = list(page.rect)
                table_dict['document_table_index'] = table_index
                table_dict['structure_id'] = "{}_{}".format(table_dict['document_id'], table_dict['document_table_index'])

                merged = False
                debug = False

                remove_empty_columns(table_dict)
                merge_columns(table_dict)      
                remove_empty_rows(table_dict)
                merge_rows(table_dict)
                        
                include = []
                exclude = []
                annotate_projected_row_headers(table_dict)

                correct_header(table_dict, assume_header_if_more_than_two_columns=True)   

                annotate_projected_row_headers(table_dict)
                
                # Look for tables with multiple headers
                num_rows = len(table_dict['rows'])
                num_columns = len(table_dict['columns'])
                cell_grid = np.zeros((num_rows, num_columns)).astype('str').tolist()
                for cell in table_dict['cells']:
                    for row_num in cell['row_nums']:
                        for column_num in cell['column_nums']:
                            cell_grid[row_num][column_num] = cell['text_content']
                for row_num1 in range(num_rows-1):
                    row1 = table_dict['rows'][row_num1]
                    if not row1['is_column_header']:
                        continue
                    for row_num2 in range(row_num1+1, num_rows):
                        row2 = table_dict['rows'][row_num2]
                        if row2['is_column_header']:
                            continue
                        if cell_grid[row_num1] == cell_grid[row_num2]:
                            print('multiple column headers')
                            #table_dict['reject'].append("multiple column headers")
                for cell1 in table_dict['cells']:
                    for cell2 in table_dict['cells']:
                        if cell1['is_column_header'] and not cell2['is_column_header']:
                            if cell1['text_content'] == cell2['text_content'] and len(cell1['text_content'].strip()) > 0:
                                #table_dict['reject'].append("multiple column headers")
                                print('multiple column headers')

                first_column_merge_exclude = []
                canonicalize(table_dict)

                remove_empty_columns(table_dict)
                merge_columns(table_dict)      
                remove_empty_rows(table_dict)
                merge_rows(table_dict)

                for row_num, row in table_dict['rows'].items():
                    if row['is_column_header'] and row_num > 4:
                        table_dict['reject'].append("oversized header")

                # Iterative process because a grid cell bounding box depends on surrounding text, which can
                # change the bounding box for the cell, which can change the text that falls in the bounding box,
                # which can change the bounding boxes for other cells, and so on...
                adjust_text = True
                iterations = 0
                while(adjust_text and iterations < 3):
                    #look_for_dots_in_text_tight_bbox(table_dict, page_words, threshold=0.5)
                    complete_table_grid(table_dict)
                    adjust_text = extract_pdf_text(table_dict, page_words)
                    iterations += 1
                if adjust_text:
                    table_dict['reject'].append("runaway text adjustment")

                num_rows = len(table_dict['rows'])
                num_cells_in_last_row = 0
                for cell in table_dict['cells']:
                    if num_rows-1 in cell['row_nums']:
                        num_cells_in_last_row += 1
                        
                # Do manual visual inspection for box-text fit
                quality_control(table_dict, page_words)

                has_body = False
                for row_num, row in table_dict['rows'].items():
                    if not row['is_column_header']:
                        has_body = True
                        break
                if not has_body:
                    table_dict['reject'].append("no table body")

                #if table_dict['rows'][0]['is_projected_row_header']:
                #    table_dict['reject'].append("bad projected row header")
                num_rows = len(table_dict['rows'])
                if table_dict['rows'][num_rows-1]['is_projected_row_header']:
                    table_dict['reject'].append("bad projected row header")
            except KeyboardInterrupt:
                do_break = True
                break
            except:
                print(traceback.format_exc())
                table_dict['reject'].append('unknown exception')
                print('not ok')
                #continue
                
            processed_count += 1

            if len(table_dict['reject']) > 0:
                reject_count += 1

                for reject_reason in set(table_dict['reject']):
                    reject_reasons[reject_reason].append(table_dict['structure_id'])

                table_dict['exclude_for_detection'] = True
                table_dict['exclude_for_structure'] = True
            else:
                good_count += 1

                if len(table_dict['fix']) > 0:
                    for fix in set(table_dict['fix']):
                        fixes[fix].append(table_dict['structure_id'])
                else:
                    kept_as_is_count += 1
                    
                document_tables.append(table_dict)

            del table_dict['reject']
            del table_dict['fix']
            
            if do_break:
                break
        
        if do_break:
            break
            
        # If not all tables present and included for detection, then exclude all for detection
        if not sum([1 for elem in document_tables if not elem['exclude_for_detection']]) == len(tables):
            for table_dict in document_tables:
                table_dict['exclude_for_detection'] = True
                
        if len(document_tables) == 0:
            continue
            
        save_filename = pdf_filepath.split("/")[-1].replace(".pdf", "") + "_tables.json"
        save_filepath = os.path.join(output_json_directory, save_filename)
        with open(save_filepath, 'w') as out_file:
            json.dump(document_tables, out_file, ensure_ascii=False, indent=4)
            
        # Create detection PASCAL VOC data
        if not document_tables[0]['exclude_for_detection']:
            detection_boxes_by_page = defaultdict(list)     
            
            # Each table has associated bounding boxes
            for table_dict in document_tables:
                try:
                    table_boxes = []

                    page_num = table_dict['pdf_page_index']

                    # Create detection data
                    class_label = 'table'
                    dict_entry = {'class_label': class_label, 'bbox': table_dict['pdf_table_bbox']}
                    detection_boxes_by_page[page_num].append(dict_entry)
                except Exception as err:
                    print(traceback.format_exc())

            # Create detection PASCAL VOC XML file and page image
            for page_num, boxes in detection_boxes_by_page.items():
                try:
                    page_bbox = table_dict['pdf_full_page_bbox']
                    if not all([is_good_bbox(entry['bbox'], page_bbox) for entry in boxes]):
                        raise Exception("At least one bounding box has non-positive area or is outside of image")

                    # Create page image
                    document_id = table_dict['document_id']
                    image_filename = document_id + "_" + str(page_num) + ".jpg"
                    image_filepath = os.path.join(output_detection_directory, "images", image_filename)
                    page_img = create_document_page_image(doc, page_num, output_image_max_dim=output_image_max_dim)

                    # Initialize PASCAL VOC XML
                    page_annotation = create_pascal_voc_page_element(image_filename, page_img.width, page_img.height,
                                                                    database="SciTSR.c-Detection")

                    for entry in boxes:
                        bbox = entry['bbox']

                        # Add to PASCAl VOC
                        element = create_pascal_voc_object_element(entry['class_label'],
                                                                entry['bbox'], page_bbox,
                                                                output_image_max_dim=output_image_max_dim)
                        page_annotation.append(element)   

                    xml_filename = document_id + "_" + str(page_num) + ".xml"
                    xml_filepath = os.path.join(output_detection_directory, split, xml_filename)

                    # Page words
                    # output_page_words_directory
                    page_rect = list(doc[page_num].rect)
                    scale = output_image_max_dim / max(page_rect)
                    tokens = []
                    for word_num, word in enumerate(doc[page_num].get_text_words()):
                        token = {}
                        token['flags'] = 0
                        token['span_num'] = word_num
                        token['line_num'] = 0
                        token['block_num'] = 0
                        bbox = [round(scale * v, 5) for v in word[:4]]
                        if Rect(bbox).get_area() > 0 and overlap(bbox, page_rect) > 0.75:
                            bbox = [max(0, bbox[0]),
                                    max(0, bbox[1]),
                                    min(page_rect[2], bbox[2]),
                                    min(page_rect[3], bbox[3])]
                            if Rect(bbox).get_area() > 0:
                                token['bbox'] = bbox
                                token['text'] = word[4]
                                tokens.append(token)

                    words_save_filepath = os.path.join(output_page_words_directory, document_id + "_" + str(page_num) + "_words.json")
                    
                    # Save
                    page_img.save(image_filepath)
                    save_xml_pascal_voc(page_annotation, xml_filepath)
                    with open(words_save_filepath, 'w', encoding='utf8') as f:
                        json.dump(tokens, f)
                except:
                    print("Exception; skipping page")
                    pass
        
        # Create structure PASCAL VOC data
        # output_structure_directory
        for table_dict in document_tables:
            if table_dict['exclude_for_structure']:
                continue
            page_num = table_dict['pdf_page_index']
            page_rect = list(doc[page_num].rect)
            scale = output_image_max_dim / max(page_rect)
            page_img = create_document_page_image(doc, page_num, output_image_max_dim=output_image_max_dim)
            
            table_num = table_dict['document_table_index']
            table_boxes = []      

            # Create structure recognition data
            class_label = 'table'
            dict_entry = {'class_label': class_label, 'bbox': table_dict['pdf_table_bbox']}
            table_boxes.append(dict_entry)

            rows = table_dict['rows'].values()
            rows = sorted(rows, key=lambda k: k['pdf_row_bbox'][1]) 
            if len(rows) > 1:
                for row1, row2 in zip(rows[:-1], rows[1:]):
                    mid_point = (row1['pdf_row_bbox'][3] + row2['pdf_row_bbox'][1]) / 2
                    row1['pdf_row_bbox'][3] = mid_point
                    row2['pdf_row_bbox'][1] = mid_point
            columns = table_dict['columns'].values()
            columns = sorted(columns, key=lambda k: k['pdf_column_bbox'][0]) 
            for col1, col2 in zip(columns[:-1], columns[1:]):
                mid_point = (col1['pdf_column_bbox'][2] + col2['pdf_column_bbox'][0]) / 2
                col1['pdf_column_bbox'][2] = mid_point
                col2['pdf_column_bbox'][0] = mid_point
            for cell in table_dict['cells']:
                column_nums = cell['column_nums']
                row_nums = cell['row_nums']
                column_rect = Rect()
                row_rect = Rect()
                for column_num in column_nums:
                    column_rect.include_rect(columns[column_num]['pdf_column_bbox'])
                for row_num in row_nums:
                    row_rect.include_rect(rows[row_num]['pdf_row_bbox'])
                cell_rect = column_rect.intersect(row_rect)
                cell['pdf_bbox'] = list(cell_rect)

            header_rect = Rect()
            for cell in table_dict['cells']:
                cell_bbox = cell['pdf_bbox']
                is_blank = len(cell['text_content'].strip()) == 0
                is_spanning_cell = len(cell['row_nums']) > 1 or len(cell['column_nums']) > 1
                is_column_header = cell['is_column_header']
                is_projected_row_header = cell['is_projected_row_header']
                if is_projected_row_header:
                    dict_entry = {'class_label': 'table projected row header', 'bbox': cell['pdf_bbox']}
                    table_boxes.append(dict_entry)                      
                elif is_spanning_cell and not is_blank:
                    dict_entry = {'class_label': 'table spanning cell', 'bbox': cell['pdf_bbox']}
                    table_boxes.append(dict_entry)                     

                if is_column_header:
                    header_rect.include_rect(cell_bbox)

            if header_rect.get_area() > 0:
                dict_entry = {'class_label': 'table column header', 'bbox': list(header_rect)}
                table_boxes.append(dict_entry)

            for row in rows:
                row_bbox = row['pdf_row_bbox']
                dict_entry = {'class_label': 'table row', 'bbox': row_bbox}
                table_boxes.append(dict_entry) 

            # table_entry['columns']
            for column in columns:
                dict_entry = {'class_label': 'table column', 'bbox': column['pdf_column_bbox']}
                table_boxes.append(dict_entry) 

            # Crop
            table_bbox = table_dict['pdf_table_bbox']

            # Convert to image coordinates
            crop_bbox = [int(round(scale * elem)) for elem in table_bbox]

            split = table_dict['split']
            if split == 'val' or split == 'test':
                padding = args.test_padding
            else:
                padding = args.train_padding
            
            # Pad
            crop_bbox = [crop_bbox[0]-padding,
                        crop_bbox[1]-padding,
                        crop_bbox[2]+padding,
                        crop_bbox[3]+padding]

            # Keep within image
            crop_bbox = [max(0, crop_bbox[0]),
                        max(0, crop_bbox[1]),
                        min(page_img.size[0], crop_bbox[2]),
                        min(page_img.size[1], crop_bbox[3])]

            table_img = page_img.crop(crop_bbox)                    
            for entry in table_boxes:
                bbox = entry['bbox']
                bbox = [scale*elem for elem in bbox]
                bbox = [max(0, bbox[0]-crop_bbox[0]-1),
                        max(0, bbox[1]-crop_bbox[1]-1),
                        min(table_img.size[0], bbox[2]-crop_bbox[0]-1),
                        min(table_img.size[1], bbox[3]-crop_bbox[1]-1)]
                entry['bbox'] = bbox

            # Initialize PASCAL VOC XML
            table_image_filename = document_id + "_table_" + str(table_num) + ".jpg"
            table_image_filepath = os.path.join(output_structure_directory, "images", table_image_filename)
            table_annotation = create_pascal_voc_page_element(table_image_filename, table_img.width, table_img.height,
                                                            database="SciTSR.c-Structure")

            table_img_bbox = [0, 0, table_img.width, table_img.height]
            try:
                if not all([is_good_bbox(entry['bbox'], table_img_bbox) for entry in table_boxes]):
                    raise Exception("At least one bounding box has non-positive area or is outside of image")

                for entry in table_boxes:
                    bbox = entry['bbox']
                    # Add to PASCAl VOC
                    element = create_pascal_voc_object_element(entry['class_label'],
                                                            entry['bbox'], [0, 0, table_img.size[0], table_img.size[1]],
                                                            output_image_max_dim=max(table_img.size))  
                    table_annotation.append(element)              

                xml_filename = table_image_filename.replace(".jpg", ".xml")
                xml_filepath = os.path.join(output_structure_directory, split, xml_filename)

                # Table words
                # output_table_words_directory
                tokens = []
                for word_num, word in enumerate(doc[page_num].get_text_words()):
                    token = {}
                    token['flags'] = 0
                    token['span_num'] = word_num
                    token['line_num'] = 0
                    token['block_num'] = 0
                    bbox = [round(scale * v, 5) for v in word[:4]]
                    if overlap(bbox, crop_bbox) > 0.75:
                        bbox = [max(0, bbox[0]-crop_bbox[0]-1),
                                max(0, bbox[1]-crop_bbox[1]-1),
                                min(table_img.size[0], bbox[2]-crop_bbox[0]-1),
                                min(table_img.size[1], bbox[3]-crop_bbox[1]-1)]
                        if Rect(bbox).get_area() > 0:
                            token['bbox'] = bbox
                            token['text'] = word[4]
                            tokens.append(token)
                        else:
                            print("REMOVED BAD TABLE WORD")

                words_save_filepath = os.path.join(output_table_words_directory, table_image_filename.replace(".jpg", "_words.json"))
                
                # Save everything
                table_img.save(table_image_filepath)
                save_xml_pascal_voc(table_annotation, xml_filepath)
                print(xml_filepath)
                with open(words_save_filepath, 'w', encoding='utf8') as f:
                    json.dump(tokens, f)
                    
            except:
                print("Exception; skipping table")
                pass
        
        del doc # Just removes from memory, not from disk

if __name__ == "__main__":
    main()