"""
Copyright (C) 2023 Microsoft Corporation

Script to edit, filter, and canonicalize FinTabNet to align it with PubTables-1M.

If you use this code in your published work, we request that you cite our papers
and table-transformer GitHub repo.
"""

import json
import os
from collections import defaultdict
import traceback
from difflib import SequenceMatcher
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse

import fitz
from fitz import Rect
from PIL import Image
import numpy as np
from tqdm import tqdm
import editdistance

# Can be used for interrupting after a specific event occurs for debugging
class DebugException(Exception):
    pass
    
    
def string_similarity(string1, string2):
    return SequenceMatcher(None, string1, string2).ratio()


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


def parse_html_table(table_html):
    try:
        tree = ET.fromstring(table_html)
    except Exception as e:
        print(e)
        return None
    
    table_cells = []
    
    occupied_columns_by_row = defaultdict(set)
    current_row = -1

    # Get all td tags
    stack = []
    stack.append((tree, False))
    while len(stack) > 0:
        current, in_header = stack.pop()

        if current.tag == 'tr':
            current_row += 1
            
        if current.tag == 'td' or current.tag =='th':
            if "colspan" in current.attrib:
                colspan = int(current.attrib["colspan"])
            else:
                colspan = 1
            if "rowspan" in current.attrib:
                rowspan = int(current.attrib["rowspan"])
            else:
                rowspan = 1
            row_nums = list(range(current_row, current_row + rowspan))
            try:
                max_occupied_column = max(occupied_columns_by_row[current_row])
                current_column = min(set(range(max_occupied_column+2)).difference(occupied_columns_by_row[current_row]))
            except:
                current_column = 0
            column_nums = list(range(current_column, current_column + colspan))
            for row_num in row_nums:
                occupied_columns_by_row[row_num].update(column_nums)
                
            cell_dict = dict()
            cell_dict['row_nums'] = row_nums
            cell_dict['column_nums'] = column_nums
            cell_dict['is_column_header'] = current.tag == 'th' or in_header
            table_cells.append(cell_dict)

        children = list(current)
        for child in children[::-1]:
            stack.append((child, in_header or current.tag == 'th' or current.tag == 'thead'))
    
    return table_cells


def create_table_dict(annotation_data):
    table_dict = {}
    table_dict['reject'] = []
    table_dict['fix'] = []
    
    html = ''.join(annotation_data['html']['structure']['tokens'])
    
    cells = parse_html_table(html)
    pdf_cells = annotation_data['html']['cells']
    
    # Make sure there are the same number of annotated HTML and PDF cells
    if not len(cells) == len(pdf_cells):
        table_dict['reject'].append("annotation mismatch")
    for cell, pdf_cell in zip(cells, pdf_cells):
        cell['json_text_content'] = ''.join(pdf_cell['tokens']).strip()
        if 'bbox' in pdf_cell:
            cell['pdf_text_tight_bbox'] = pdf_cell['bbox']
        else:
            cell['pdf_text_tight_bbox'] = []
        
    # Make sure no grid locations are duplicated
    grid_cell_locations = []
    for cell in cells:
        for row_num in cell['row_nums']:
            for column_num in cell['column_nums']:
                grid_cell_locations.append((row_num, column_num))
    if not len(grid_cell_locations) == len(set(grid_cell_locations)):
        table_dict['reject'].append("HTML overlapping grid cells")
        
    grid_cell_locations = set(grid_cell_locations)
                
    num_rows = max([max(cell['row_nums']) for cell in cells]) + 1
    num_columns = max([max(cell['column_nums']) for cell in cells]) + 1
    expected_num_cells = num_rows * num_columns
    actual_num_cells = len(grid_cell_locations)
        
    # Make sure all grid locations are present
    if not expected_num_cells == actual_num_cells:
        table_dict['reject'].append("HTML missing grid cells")
        
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
        
    # Intersect each row and column to determine grid cell bounding boxes
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
        if len(cell['json_text_content']) > 0:
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
            if len(cell['json_text_content']) > 0:
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
        cell['json_text_content'] = (cell['json_text_content'].strip() + " " + cell2['json_text_content'].strip()).strip()
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
        
        
def remove_empty_rows(table_dict):
    num_rows = len(table_dict['rows'])
    num_columns = len(table_dict['columns'])
    has_content_by_row = defaultdict(bool)
    for cell in table_dict['cells']:
        has_content = len(cell['json_text_content'].strip()) > 0
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
        has_content = len(cell['json_text_content'].strip()) > 0
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
    numeric_count_by_column = defaultdict(int)
    alpha_count_by_column = defaultdict(int)
    for cell in table_dict['cells']:
        if cell['is_column_header'] or cell['is_projected_row_header']:
            continue
        numeric_count = sum([1 for ch in cell['json_text_content'] if ch.isnumeric()])
        alpha_count = sum([1 for ch in cell['json_text_content'] if ch.isalpha()])
        for column_num in cell['column_nums']:
            numeric_count_by_column[column_num] += numeric_count
            alpha_count_by_column[column_num] += alpha_count
    if not alpha_count_by_column[1] > numeric_count_by_column[1]:
        return

    first_column_cells = [cell for cell in table_dict['cells'] if 0 in cell['column_nums']]
    first_column_cells = sorted(first_column_cells, key=lambda item: max(item['row_nums']))
    
    current_filled_cell = None
    groups = defaultdict(list)
    group_num = -1
    for cell in first_column_cells:
        if len(cell['json_text_content']) > 0:
            group_num += 1
        if group_num >= 0:
            groups[group_num].append(cell)
        
    for group_num, group in groups.items():
        if len(group) > 1 and not group[0]['is_projected_row_header'] and not group[0]['is_column_header']:
            merge_group(table_dict, group)
            
            
def correct_header(table_dict, assume_header_if_more_than_two_columns=True):
    num_columns = len(table_dict['columns'])
    num_rows = len(table_dict['rows'])
    
    if num_columns < 2 or num_rows < 1:
        table_dict['reject'].append("small table")
        
    #---DETERMINE FULL EXTENT OF COLUMN HEADER
    # - Each of the below steps determines different rows that must be in the column header.
    # - The final column header includes all rows that are originally annotated as being in the column
    #   header plus any additional rows determined to be in the column header by the following steps.
    
    table_has_column_header = False
    
    # First determine if there is definitely a column header. Cases:
    
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
            if 0 in cell['column_nums'] and 0 in cell['row_nums'] and len(cell['json_text_content'].strip()) == 0:
                table_has_column_header = True
                break
    
    # 4. There is a horizontal spanning cell in the first row
    if not table_has_column_header:
        for cell in table_dict['cells']:
            if 0 in cell['row_nums'] and len(cell['column_nums']) > 1:
                table_has_column_header = True
                break
                
    # 5. Particular words or phrases appear in the first row 
    if not table_has_column_header:
        for cell in table_dict['cells']:
            if 0 in cell['row_nums'] and 0 in cell['column_nums'] and 'Number' in cell['json_text_content']:
                table_dict['fix'].append("two column header: Number")
                table_has_column_header = True
                break
            if 0 in cell['row_nums'] and 1 in cell['column_nums'] and 'Page' in cell['json_text_content']:
                table_dict['fix'].append("two column header: Page")
                table_has_column_header = True
                break
            if 0 in cell['row_nums'] and 'in thousands' in cell['json_text_content'].lower():
                table_dict['fix'].append("two column header: in thousands")
                table_has_column_header = True
                break
            if 0 in cell['row_nums'] and 'in millions' in cell['json_text_content'].lower():
                table_dict['fix'].append("two column header: in millions")
                table_has_column_header = True
                break
            if 0 in cell['row_nums'] and 'Measurement' in cell['json_text_content']:
                table_dict['fix'].append("two column header: Measurement")
                table_has_column_header = True
                break
            if 0 in cell['row_nums'] and 'Period' in cell['json_text_content']:
                table_dict['fix'].append("two column header: Period")
                table_has_column_header = True
                break

    # Then determine if the column header needs to be extended past its current annotated extent.
    #  1. A header that already is annotated in at least one row continues at least until each column
    #     has a cell occupying only that column
    #  2. A header with a column with a blank cell must continue at least as long as the blank cells continue
    #     (unless rule #1 is satisfied and a possible projected row header is reached?)
    if table_has_column_header:
        first_column_filled_by_row = defaultdict(bool)
        for cell in table_dict['cells']:
            if 0 in cell['column_nums']:
                if len(cell['json_text_content']) > 0:
                    for row_num in cell['row_nums']:
                        first_column_filled_by_row[row_num] = True
        
        first_single_node_row_by_column = defaultdict(lambda: len(table_dict['rows'])-1)
        for cell in table_dict['cells']:
            if len(cell['column_nums']) == 1:
                first_single_node_row_by_column[cell['column_nums'][0]] = min(first_single_node_row_by_column[cell['column_nums'][0]],
                                                                               max(cell['row_nums']))
                
        first_filled_single_node_row_by_column = defaultdict(lambda: len(table_dict['rows'])-1)
        for cell in table_dict['cells']:
            if len(cell['column_nums']) == 1 and len(cell['json_text_content'].strip()) > 0:
                first_filled_single_node_row_by_column[cell['column_nums'][0]] = min(first_filled_single_node_row_by_column[cell['column_nums'][0]],
                                                                               max(cell['row_nums']))
                
        first_filled_cell_by_column = defaultdict(lambda: len(table_dict['rows'])-1)
        for cell in table_dict['cells']:
            if len(cell['json_text_content']) > 0:
                min_row_num = min(cell['row_nums'])
                for column_num in cell['column_nums']:
                    first_filled_cell_by_column[column_num] = min(first_filled_cell_by_column[column_num],
                                                                  min_row_num)
                    
        projected_row_header_rows = identify_projected_row_headers(table_dict)
        if 0 in projected_row_header_rows:
            table_dict['reject'].append("bad projected row header")
        
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

        if len(projected_row_header_rows) > 0:
            minimum_projected_row_header_row = min(projected_row_header_rows)
        else:
            minimum_projected_row_header_row = num_rows

        first_possible_last_header_row = minimum_first_body_row - 1
                    
        last_header_row = max(minimum_all_following_filled,
                              minimum_grid_cell_single_node_row,
                              first_possible_last_header_row)
        
        x = last_header_row
        while(last_header_row < num_rows and not first_column_filled_by_row[last_header_row+1]):
            last_header_row += 1            
        
        if minimum_projected_row_header_row <= last_header_row:
            last_header_row = minimum_projected_row_header_row - 1
        
        for cell in table_dict['cells']:
            if max(cell['row_nums']) <= last_header_row:
                cell['is_column_header'] = True
        
        for row_num, row in table_dict['rows'].items():
            if row_num <= last_header_row:
                row['is_column_header'] = True
    
    if not table_has_column_header and num_columns == 2:
        keep_table = False
        for cell in table_dict['cells']:
            if 0 in cell['row_nums'] and len(cell['json_text_content']) > 60:
                keep_table = True
                table_dict['fix'].append("two column no header: long text")
                break
            if 0 in cell['row_nums'] and 1 in cell['column_nums'] and re.match('^[0-9,%\.\$ -]+$', cell['json_text_content']):
                keep_table = True
                table_dict['fix'].append("two column no header: numeric")
                break
        
        if not keep_table:
            table_dict['reject'].append("ambiguous header")

def canonicalize(table_dict):
    # Preprocessing step: Split every blank spanning cell in the column header into blank grid cells.
    cells_to_delete = []
    try:
        for cell in table_dict['cells']:
            if (cell['is_column_header'] and len(cell['json_text_content'].strip()) == 0
                    and (len(cell['column_nums']) > 1 or len(cell['row_nums']) > 1)):
                cells_to_delete.append(cell)
                # Split this blank spanning cell into blank grid cells
                for column_num in cell['column_nums']:
                    for row_num in cell['row_nums']:
                        new_cell = {'json_text_content': '',
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
        if not cell['is_column_header'] or len(cell['json_text_content']) == 0:
            continue
        header_group = [cell]
        next_row_num = min(cell['row_nums']) - 1
        for row_num in range(next_row_num, -1, -1):
            all_are_blank = True
            for column_num in cell['column_nums']:
                cell2 = cell_grid_index[(row_num, column_num)]
                all_are_blank = all_are_blank and len(cell2['json_text_content']) == 0
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
        if not cell['is_column_header'] or len(cell['json_text_content']) == 0:
            continue
        header_group = [cell]
        next_row_num = max(cell['row_nums']) + 1
        for row_num in range(next_row_num, num_rows):
            if not table_dict['rows'][row_num]['is_column_header']:
                break
            all_are_blank = True
            for column_num in cell['column_nums']:
                cell2 = cell_grid_index[(row_num, column_num)]
                all_are_blank = all_are_blank and len(cell2['json_text_content']) == 0
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
        xml_text = ''.join(cell['json_text_content'].split()).strip('.')
        pdf_text = ''.join(cell['pdf_text_content'].split()).strip('.')
        L = max(len(xml_text), len(pdf_text))
        if L > 0:
            D += editdistance.eval(xml_text, pdf_text) / L
            
    return D / len(cells)
        
def quality_control1(table_dict, page_words):
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
        
def quality_control2(table_dict, page_words):
    for row_num1, row1 in table_dict['rows'].items():
        for row_num2, row2, in table_dict['rows'].items():
            if row_num1 == row_num2 - 1:
                if row1['pdf_row_bbox'][3] > row2['pdf_row_bbox'][1] + 1:
                    table_dict['reject'].append("rows intersect")
                    
    for column_num1, column1 in table_dict['columns'].items():
        for column_num2, column2, in table_dict['columns'].items():
            if column_num1 == column_num2 - 1:
                if column1['pdf_column_bbox'][2] > column2['pdf_column_bbox'][0] + 1:
                    table_dict['reject'].append("columns intersect")
    
    D = table_text_edit_distance(table_dict['cells'])
    if D > 0.05:
        table_dict['reject'].append("text annotation quality")
        
        
def remove_html_tags_in_text(table_dict):
    for cell in table_dict['cells']:
        cell['json_text_content'] = cell['json_text_content'].replace("<i>", " ")
        cell['json_text_content'] = cell['json_text_content'].replace("</i>", " ")
        cell['json_text_content'] = cell['json_text_content'].replace("<sup>", " ")
        cell['json_text_content'] = cell['json_text_content'].replace("</sup>", " ")
        cell['json_text_content'] = cell['json_text_content'].replace("<sub>", " ")
        cell['json_text_content'] = cell['json_text_content'].replace("</sub>", " ")
        cell['json_text_content'] = cell['json_text_content'].replace("  ", " ")
        cell['json_text_content'] = cell['json_text_content'].strip()
        
        
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


def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1).intersect(bbox2)
    
    bbox1_area = Rect(bbox1).get_area()
    if bbox1_area > 0:
        return intersection.get_area() / bbox1_area
    
    return 0


def get_tokens_in_table_img(page_words, table_img_bbox):
    tokens = []
    for word_num, word in enumerate(page_words):
        word['flags'] = 0
        word['span_num'] = word_num
        word['line_num'] = 0
        word['block_num'] = 0
        tokens.append(word)

    tokens_in_table = [token for token in tokens if iob(token['bbox'], table_img_bbox) >= 0.5]
    
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

    pdf_directory = os.path.join(args.data_dir, "pdf")

    output_json_directory = os.path.join(args.output_dir, "FinTabNet.c_PDF_Annotations_JSON")
    if not os.path.exists(output_json_directory):
        os.makedirs(output_json_directory)

    output_subdirs = ['images', 'train', 'test', 'val']
    output_structure_directory = os.path.join(args.output_dir, "FinTabNet.c_Image_Structure_PASCAL_VOC")
    if not os.path.exists(output_structure_directory):
        os.makedirs(output_structure_directory)
    for subdir in output_subdirs:
        subdirectory = os.path.join(output_structure_directory, subdir)
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

    output_table_words_directory = os.path.join(args.output_dir, "FinTabNet.c_Image_Table_Words_JSON")
    if not os.path.exists(output_table_words_directory):
        os.makedirs(output_table_words_directory)

    # These are samples that killed the kernel during processing due to unknown error; likely an OOM issue
    # Skipping these in this script, although they could be added back in the future
    samples_to_skip = defaultdict(set)
    samples_to_skip['train'] = set([48651, 48652, 48659, 48660, 48672, 48673, 48674, 48675, 48691, 48692, 48693, 48694])

    table_count_by_document_id = defaultdict(int)
    file_idx_to_table_idx = dict()

    processed_count = 0
    accepted_count = 0
    reject_count = 0
    reject_reasons = defaultdict(list)
    fixes = defaultdict(list)
    kept_as_is_count = 0
    save_count = 0
    output_image_max_dim = 1000

    do_save = True
    do_break = False

    for subdir in ['val', 'test', 'train']:
        if subdir == 'val' or subdir == 'test':
            padding = args.test_padding
        else:
            padding = args.train_padding

        print("Processing '{}' samples...".format(subdir))
        structure_filename = "FinTabNet_1.0.0_cell_" + subdir + ".jsonl"
        detection_filename = "FinTabNet_1.0.0_table_" + subdir +  ".jsonl"
        structure_filepath = os.path.join(args.data_dir, structure_filename)
        detection_filepath = os.path.join(args.data_dir, detection_filename)

        with open(structure_filepath, "r") as f:
            structure_lines = f.readlines()
        with open(detection_filepath, "r") as f:
            detection_lines = f.readlines()

        structure_tables = defaultdict(set)
        for idx, line in enumerate(structure_lines):
            data = json.loads(line)
            structure_tables[data['filename']].add(idx)

        detection_tables = defaultdict(set)
        for line in detection_lines:
            data = json.loads(line)
            detection_tables[data['filename']].add(data['table_id'])

        table_count_by_document_id = defaultdict(int)
        file_idx_to_table_idx = dict()

        filename = structure_filename
        lines = structure_lines

        for idx, line in enumerate(lines):
            data = json.loads(lines[idx])

            document_id = "_".join(data['filename'].split(".")[0].split("/"))
            file_idx_to_table_idx[idx] = table_count_by_document_id[document_id]
            table_count_by_document_id[document_id] += 1

        file_count = 0
        for relative_pdf_filepath, idxs in tqdm(structure_tables.items()):
            file_count += 1
            if len(set(idxs).intersection(samples_to_skip[subdir])) > 0:
                print("SKIPPING {}".format(relative_pdf_filepath))
                continue
            pdf_filepath = os.path.join(pdf_directory, relative_pdf_filepath)
            save_filename = relative_pdf_filepath.replace(".pdf", "").replace("/", "_") + "_tables.json"
            save_filepath = os.path.join(output_json_directory, save_filename)

            doc = fitz.open(pdf_filepath)
            page = doc[0]
            page_words = page.get_text_words()
            page_bbox = list(page.rect)
            for w in page_words[:]:
                if Rect(w[:4]).get_area() == 0 or overlap(w[:4], page_bbox) < 1:
                    page_words.remove(w)

            document_tables = []
            for idx in idxs:
                data = json.loads(lines[idx])

                try:
                    adjust_bbox_coordinates(data, doc)
                    table_dict = create_table_dict(data)

                    exclude_for_structure = False
                    exclude_for_detection = not relative_pdf_filepath in detection_tables

                    table_dict['exclude_for_structure'] = exclude_for_structure
                    table_dict['exclude_for_detection'] = exclude_for_detection
                    table_dict['split'] = data['split']
                    table_dict['pdf_file_name'] = data['filename'].split("/")[-1]
                    table_dict['pdf_folder'] = "/".join(data['filename'].split("/")[:-1]) + "/"
                    table_dict['document_id'] = "_".join(data['filename'].split(".")[0].split("/"))
                    table_dict['fintabnet_source_file_name'] = filename
                    table_dict['fintabnet_source_line_index'] = idx
                    table_dict['fintabnet_source_table_id'] = data['table_id']
                    table_dict['pdf_page_index'] = 0
                    table_dict['pdf_full_page_bbox'] = list(page.rect)
                    table_dict['document_table_index'] = file_idx_to_table_idx[idx]  # need to create a mapping for this
                    table_dict['structure_id'] = "{}_{}".format(table_dict['document_id'], table_dict['document_table_index'])

                    # Initial fixes/adjustments
                    remove_html_tags_in_text(table_dict)

                    for cell in table_dict['cells']:
                        if 0 in cell['row_nums'] and len(cell['row_nums']) > 2 and len(cell['json_text_content'].strip()) > 0:
                            table_dict['reject'].append("overmerged cells")

                    merged = False
                    debug = False

                    remove_empty_columns(table_dict)
                    merge_columns(table_dict)      
                    remove_empty_rows(table_dict)
                    merge_rows(table_dict)

                    for cell in table_dict['cells']:
                        if cell['json_text_content'] in ['¢', '$']:
                            table_dict['reject'].append("oversegmented columns")

                    total_characters_by_column = defaultdict(int)
                    has_small_filled_cell_by_column = defaultdict(int)
                    for cell in table_dict['cells']:
                        if len(cell['column_nums']) == 1:
                            column_num = cell['column_nums'][0]
                            total_characters_by_column[column_num] += len(cell['json_text_content'])
                            if (len(cell['json_text_content']) > 0
                                and (len(cell['json_text_content']) < 2
                                     or (len(cell['json_text_content']) < 4
                                         and cell['json_text_content'][0] == '('))):
                                has_small_filled_cell_by_column[column_num] = True
                    num_rows = len(table_dict['rows'])
                    for column_num, total in total_characters_by_column.items():
                        if total < num_rows and has_small_filled_cell_by_column[column_num]:
                            table_dict['reject'].append("oversegmented columns")

                    correct_header(table_dict, assume_header_if_more_than_two_columns=True)
                    annotate_projected_row_headers(table_dict)

                    # Putting canonicalization before bounding box determination
                    canonicalize(table_dict)

                    remove_empty_columns(table_dict)
                    merge_columns(table_dict)      
                    remove_empty_rows(table_dict)
                    merge_rows(table_dict)

                    num_columns = len(table_dict['columns'])
                    for row_num, row in table_dict['rows'].items():
                        if row['is_column_header'] and (row_num > 4 or row_num >= num_columns-1):
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

                    for word in page_words:
                        cell_set = set()
                        for cell_num, cell in enumerate(table_dict['cells']):
                            if iob(word[:4], cell['pdf_bbox']) >= 0.5:
                                cell_set.add(cell_num)
                        if len(cell_set) > 1:
                            table_dict['reject'].append('overlapping cells')
                            break

                    # Filter out inconsistent tables, unusual tables, and tables with
                    # potentially low annotation quality

                    quality_control1(table_dict, page_words)

                    has_body = False
                    for row_num, row in table_dict['rows'].items():
                        if not row['is_column_header']:
                            has_body = True
                            break
                    if not has_body:
                        table_dict['reject'].append("no table body")

                    if table_dict['rows'][0]['is_projected_row_header']:
                        table_dict['reject'].append("bad projected row header")
                    num_rows = len(table_dict['rows'])
                    if table_dict['rows'][num_rows-1]['is_projected_row_header']:
                        table_dict['reject'].append("bad projected row header")

                    # Check that everything is properly contained
                    table_bbox = table_dict['pdf_table_bbox']
                    for cell in table_dict['cells']:
                        bbox = cell['pdf_bbox']
                        if (Rect(bbox).get_area() == 0 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]
                            or overlap(bbox, page_bbox) < 1 or overlap(bbox, table_bbox) < 1):
                            table_dict['reject'].append("bad cell bbox")
                            raise Exception("Bad cell bbox: {}".format(bbox))

                    if (Rect(bbox).get_area() == 0 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]
                        or overlap(table_bbox, page_bbox) < 1):
                        table_dict['reject'].append("bad table bbox")
                        raise Exception("Bad table bbox: {}".format(table_dict['pdf_table_bbox']))
                except KeyboardInterrupt:
                    do_break = True
                    break
                except:
                    #print(idx)
                    #print(traceback.format_exc())
                    table_dict['reject'].append('unknown exception')

                processed_count += 1

                if len(table_dict['reject']) > 0:
                    reject_count += 1

                    for reject_reason in set(table_dict['reject']):
                        reject_reasons[reject_reason].append(idx)

                    table_dict['exclude_for_detection'] = True
                    table_dict['exclude_for_structure'] = True
                else:
                    accepted_count += 1

                    if len(table_dict['fix']) > 0:
                        for fix in set(table_dict['fix']):
                            fixes[fix].append(idx)
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
            if not sum([1 for elem in document_tables if not elem['exclude_for_detection']]) == len(idxs):
                for table_dict in document_tables:
                    table_dict['exclude_for_detection'] = True

            if len(document_tables) == 0:
                continue

            if do_save:
                with open(save_filepath, 'w') as out_file:
                    json.dump(document_tables, out_file, ensure_ascii=False, indent=4)

            # Create structure PASCAL VOC data
            # output_structure_directory
            for table_dict in document_tables:
                split = table_dict['split']
                document_id = table_dict['document_id']
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
                    is_blank = len(cell['json_text_content'].strip()) == 0
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
                bad_box = False
                for entry in table_boxes:
                    bbox = entry['bbox']
                    bbox = [scale*elem for elem in bbox]
                    bbox = [max(0, bbox[0]-crop_bbox[0]-1),
                            max(0, bbox[1]-crop_bbox[1]-1),
                            min(table_img.size[0], bbox[2]-crop_bbox[0]-1),
                            min(table_img.size[1], bbox[3]-crop_bbox[1]-1)]
                    if (bbox[0] < 0 or bbox[1] < 0 or bbox[2] > table_img.size[0] or bbox[3] > table_img.size[1]
                        or bbox[0] + 1 > bbox[2] or bbox[1] + 1 > bbox[3]):
                        bad_box = True
                    entry['bbox'] = bbox

                if bad_box:
                    print("BAD BOX, SKIPPING TABLE")
                    continue

                # Initialize PASCAL VOC XML
                table_image_filename = document_id + "_table_" + str(table_num) + ".jpg"
                table_image_filepath = os.path.join(output_structure_directory, "images", table_image_filename)
                table_annotation = create_pascal_voc_page_element(table_image_filename, table_img.width, table_img.height,
                                                                  database="FinTabNet.c-Structure")


                for entry in table_boxes:
                    bbox = entry['bbox']
                    # Add to PASCAl VOC
                    element = create_pascal_voc_object_element(entry['class_label'],
                                                               entry['bbox'], [0, 0, table_img.size[0], table_img.size[1]],
                                                               output_image_max_dim=max(table_img.size))  
                    table_annotation.append(element)              

                if do_save:
                    table_img.save(table_image_filepath)

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
                        if (bbox[0] < 0 or bbox[1] < 0 or bbox[2] > table_img.size[0] or bbox[3] > table_img.size[1]
                            or bbox[0] > bbox[2] or bbox[1] > bbox[3]):
                            bad_box = True
                        else:
                            token['bbox'] = bbox
                            token['text'] = word[4]
                            tokens.append(token)

                words_save_filepath = os.path.join(output_table_words_directory, table_image_filename.replace(".jpg", "_words.json"))

                if do_save:
                    save_xml_pascal_voc(table_annotation, xml_filepath)
                    with open(words_save_filepath, 'w', encoding='utf8') as f:
                        json.dump(tokens, f)
                    save_count += 1

            del doc

    print("-------------------------------------------------------------------")
    print(" REPORT:")
    print("-------------------------------------------------------------------")
    
    reject_counts = defaultdict(int)
    reject_counts.update({k: len(v) for k, v in reject_reasons.items()})

    print("Correction type counts:")
    for reason, idxs in fixes.items():
        print("{}: {}".format(reason, len(idxs)))
    print("-------")
    print("Rejection reason counts:")
    for reason, count in reject_counts.items():
        print("{}: {}".format(reason, count))
    print("-------")
    print("Final summary:")
    print("{} processed tables".format(processed_count))
    print("{} rejected tables".format(reject_count))
    print("{} accepted tables".format(accepted_count))
    print("{} adjusted tables".format(accepted_count - kept_as_is_count))
    print("{} non-adjusted tables".format(kept_as_is_count))
    print("{} saved tables".format(save_count))

if __name__ == "__main__":
    main()