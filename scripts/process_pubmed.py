"""
Copyright (C) 2023 Microsoft Corporation

USAGE NOTES:
This code is our best attempt to piece together the code that was used to create PubTables-1M.
(PubTables-1M was originally created in multiple stages, not all in one script.)

This script processes pairs of PDF and NXML files in the PubMed Open Access corpus.

These need to be downloaded first.
Download tar.gz files from the FTP site:
https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/

Please pay attention to licensing. See the PubMed Central website for more information on
how to ensure you download data licensed for commercial use, if that is your need.

Before running this script, place the downloaded files in the same directory and unzip them.
This should create a collection of subdirectories each starting with "PMC..." like so:
parent_folder\
- PMC1234567\
  - same_name.pdf
  - same_name.nxml
- PMC2345678\
- PMC3456789\

Note that this script has a timeout for each file and skips ones that take too long to process.

If you use this code in your published work, we ask that you please cite our PubTables-1M paper
and table-transformer GitHub repo.

TODO:
- Add code for making or incorporating a train/test/val split
- Change the table padding for the test and val splits
"""

import os
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
import json
import functools
from collections import defaultdict
import traceback
import signal
import argparse

from PIL import Image
import numpy as np
import fitz
from fitz import Rect
import editdistance

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def read_xml(nxml_filepath):
    '''
    Read in XML as a string.
    '''
    with open(nxml_filepath, 'r') as file:
        xml_string = file.read()
        
    return xml_string


def read_pdf(pdf_filepath):
    '''
    Read in PDF file as a PyMyPDF doc.
    '''
    doc = fitz.open(pdf_filepath)
    
    return doc


def compare_meta(word1, word2):
    '''
    For ordering words according to *some* reading order within the PDF.
    '''
    if word1[5] < word2[5]:
        return -1
    if word1[5] > word2[5]:
        return 1

    if word1[6] < word2[6]:
        return -1
    if word1[6] > word2[6]:
        return 1
    
    if word1[7] < word2[7]:
        return -1
    if word1[7] > word2[7]:
        return 1

    return 0


def get_page_words(page):
    """
    Extract the words from the page, with bounding boxes,
    as well as loose layout and style information.
    """
    words = []

    for text_word in page.get_text_words():
        word = {'bbox': list(text_word[:4]),
                'text': text_word[4],
                'block_num': text_word[5],
                'line_num': text_word[6],
                'span_num': text_word[7],
                'flags': 0}
        words.append(word)

    return words

def overlaps(bbox1, bbox2, threshold=0.5):
    """
    Test if more than "threshold" fraction of bbox1 overlaps with bbox2.
    """
    rect1 = Rect(bbox1)
    area1 = rect1.get_area()
    if area1 == 0:
        return False
    return rect1.intersect(bbox2).get_area()/area1 >= threshold

def get_bbox_span_subset(spans, bbox, threshold=0.5):
    """
    Reduce the set of spans to those that fall within a bounding box.

    threshold: the fraction of the span that must overlap with the bbox.
    """
    span_subset = []
    for span in spans:
        if overlaps(span['bbox'], bbox, threshold):
            span_subset.append(span)
    return span_subset


def extract_text_from_spans(spans, join_with_space=True, remove_integer_superscripts=True):
    """
    Convert a collection of page tokens/words/spans into a single text string.
    """

    if join_with_space:
        join_char = " "
    else:
        join_char = ""
    spans_copy = spans[:]
    
    if remove_integer_superscripts:
        for span in spans:
            flags = span['flags']
            if flags & 2**0: # superscript flag
                if is_int(span['text']):
                    spans_copy.remove(span)
                else:
                    span['superscript'] = True

    if len(spans_copy) == 0:
        return ""
    
    spans_copy.sort(key=lambda span: span['span_num'])
    spans_copy.sort(key=lambda span: span['line_num'])
    spans_copy.sort(key=lambda span: span['block_num'])
    
    # Force the span at the end of every line within a block to have exactly one space
    # unless the line ends with a space or ends with a non-space followed by a hyphen
    line_texts = []
    line_span_texts = [spans_copy[0]['text']]
    for span1, span2 in zip(spans_copy[:-1], spans_copy[1:]):
        if not span1['block_num'] == span2['block_num'] or not span1['line_num'] == span2['line_num']:
            line_text = join_char.join(line_span_texts).strip()
            if (len(line_text) > 0
                    and not line_text[-1] == ' '
                    and not (len(line_text) > 1 and line_text[-1] == "-" and not line_text[-2] == ' ')):
                if not join_with_space:
                    line_text += ' '
            line_texts.append(line_text)
            line_span_texts = [span2['text']]
        else:
            line_span_texts.append(span2['text'])
    line_text = join_char.join(line_span_texts)
    line_texts.append(line_text)
            
    return join_char.join(line_texts).strip()


def extract_text_inside_bbox(spans, bbox):
    """
    Extract the text inside a bounding box.
    """
    bbox_spans = get_bbox_span_subset(spans, bbox)
    bbox_text = extract_text_from_spans(bbox_spans, remove_integer_superscripts=False)

    return bbox_text, bbox_spans


def extract_table_xmls_from_document(xml_string):
    table_dicts = []

    table_starts = [m.start() for m in re.finditer("<table-wrap |<table-wrap>", xml_string)]
    table_ends = [m.end() for m in re.finditer("</table-wrap>", xml_string)]
    if not len(table_starts) == len(table_ends):
        print("Could not match up all table-wrap begins and ends")
        return None

    for table_start, table_end in zip(table_starts, table_ends):
        table_dict = {}
        table_dict['xml_table_wrap_start_character_index'] = table_start
        table_dict['xml_table_wrap_end_character_index'] = table_end
        table_dicts.append(table_dict)
        
    return table_dicts


def parse_xml_table(xml_string, table_dict):
    start_index = table_dict['xml_table_wrap_start_character_index']
    end_index = table_dict['xml_table_wrap_end_character_index']
    table_xml = xml_string[start_index:end_index]
    table_dict['xml_markup'] = table_xml
    
    try:
        table_xml = table_xml.replace("xlink:", "") # these break the xml parser
        tree = ET.fromstring(table_xml)
    except Exception as e:
        print(e)
        return None
    
    table_cells = []
    
    occupied_columns_by_row = defaultdict(set)
    current_row = -1

    caption_text = []
    
    # Initialize empty values
    table_dict['xml_tablewrap_raw_text'] = ""
    table_dict['xml_table_raw_text'] = ""
    table_dict['xml_graphic_filename'] = ""
    table_dict['xml_table_footer_text'] = ""
    table_dict['xml_caption_label_text'] = ""
    table_dict['xml_caption_text'] = ""

    # Get all td tags
    stack = []
    stack.append((tree, False))
    while len(stack) > 0:
        current, in_header = stack.pop()
        if current.tag == 'table-wrap':
            table_dict['xml_tablewrap_raw_text'] = ' '.join([elem.strip() for elem in current.itertext()]).strip()
        if current.tag == 'table':
            table_dict['xml_table_raw_text'] = ' '.join([elem.strip() for elem in current.itertext()]).strip()
        if current.tag == 'graphic':
            try:
                table_dict['xml_graphic_filename'] = current.attrib['href']
            except:
                pass
        if current.tag == 'table-wrap-foot':
            table_dict['xml_table_footer_text'] = ''.join(current.itertext()).strip()
        if current.tag == 'label':
            table_dict['xml_caption_label_text'] = ''.join(current.itertext()).strip()
        if current.tag == 'caption':
            table_dict['xml_caption_text'] = ''.join(current.itertext()).strip()
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
                
            if "align" in current.attrib:
                align = current.attrib["align"]
            else:
                align = "unknown"
            if "style" in current.attrib:
                style = current.attrib["style"]
            else:
                style = "none"
            graphics = [child for child in current if child.tag == 'graphic']
            graphics_filenames = [graphic.attrib['href'] for graphic in graphics if "href" in graphic.attrib]
                
            raw_text = ''.join(current.itertext())
            text = ' '.join([elem.strip() for elem in current.itertext()])
            cell_dict = dict()
            cell_dict['row_nums'] = row_nums
            cell_dict['column_nums'] = column_nums
            cell_dict['is_column_header'] = current.tag == 'th' or in_header
            cell_dict['align'] = align
            cell_dict['style'] = style
            # tab or space or padding
            if (raw_text.startswith("\u2003") or raw_text.startswith("\u0020")
                or raw_text.startswith("\t") or raw_text.startswith(" ")
                or "padding-left" in style):
                cell_dict['indented'] = True
            else:
                cell_dict['indented'] = False
            cell_dict['xml_text_content'] = text
            cell_dict['xml_raw_text_content'] = raw_text
            cell_dict['xml_graphics_filenames'] = graphics_filenames
            #cell_dict['pdf'] = {}
            table_cells.append(cell_dict)

        children = list(current)
        for child in children[::-1]:
            stack.append((child, in_header or current.tag == 'th' or current.tag == 'thead'))

    #table_dict['rows'] = [{} for entry in range(row_num + 1)]
    #table_dict['columns'] = [{} for entry in range(num_cols)]
    
    if len(occupied_columns_by_row) > 0:
        table_dict['num_rows'] = max(occupied_columns_by_row) + 1
        table_dict['num_columns'] = max([max(elems) for row_num, elems in occupied_columns_by_row.items()]) + 1
    else:
        table_dict['num_rows'] = 0
        table_dict['num_columns'] = 0
    
    table_dict['cells'] = table_cells
    
    return table_dict


# For traceback: -1 = up, 1 = left, 0 = diag up-left

def align(page_string="", table_string="", match_reward=2, mismatch_penalty=-5, new_gap_penalty=-2,
          continue_gap_penalty=-0.05, page_boundary_gap_reward=0.01, gap_not_after_space_penalty=-1,
          score_only=False, gap_character='_'):
    
    scores = np.zeros((len(page_string) + 1, len(table_string) + 1))
    pointers = np.zeros((len(page_string) + 1, len(table_string) + 1))
    
    # Initialize first column
    for row_idx in range(1, len(page_string) + 1):
        scores[row_idx, 0] = scores[row_idx - 1, 0] + page_boundary_gap_reward
        pointers[row_idx, 0] = -1
        
    # Initialize first row
    for col_idx in range(1, len(table_string) + 1):
        #scores[0, col_idx] = scores[0, col_idx - 1] + 0
        pointers[0, col_idx] = 1
        
    for row_idx in range(1, len(page_string) + 1):
        for col_idx in range(1, len(table_string) + 1):
            if page_string[row_idx - 1] == table_string[col_idx - 1]:
                diag_score = scores[row_idx - 1, col_idx - 1] + match_reward
            else:
                diag_score = scores[row_idx - 1, col_idx - 1] + mismatch_penalty
            
            if pointers[row_idx, col_idx - 1] == 1:
                same_row_score = scores[row_idx, col_idx - 1] + continue_gap_penalty
            else:
                same_row_score = scores[row_idx, col_idx - 1] + new_gap_penalty
                if not table_string[col_idx - 1] == ' ':
                    same_row_score += gap_not_after_space_penalty
            
            if col_idx == len(table_string):
                same_col_score = scores[row_idx - 1, col_idx] + page_boundary_gap_reward
            elif pointers[row_idx - 1, col_idx] == -1:
                same_col_score = scores[row_idx - 1, col_idx] + continue_gap_penalty
            else:
                same_col_score = scores[row_idx - 1, col_idx] + new_gap_penalty
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
        
    #print(scores[:, -1])
    #print(pointers)
    
    score = scores[len(page_string), len(table_string)]
    
    if score_only:
        return score
    
    cur_row = len(page_string)
    cur_col = len(table_string)
    aligned_page_string = ""
    aligned_table_string = ""
    while not (cur_row == 0 and cur_col == 0):
        if pointers[cur_row, cur_col] == -1:
            cur_row -= 1
            aligned_table_string += gap_character
            aligned_page_string += page_string[cur_row]
        elif pointers[cur_row, cur_col] == 1:
            cur_col -= 1
            aligned_page_string += gap_character
            aligned_table_string += table_string[cur_col]
        else:
            cur_row -= 1
            cur_col -= 1
            aligned_table_string += table_string[cur_col]
            aligned_page_string += page_string[cur_row]
            
    aligned_page_string = aligned_page_string[::-1]
    aligned_table_string = aligned_table_string[::-1]
    
    alignment = [aligned_page_string, aligned_table_string]
    
    return alignment, score


def get_table_page_fast(doc, table):
    table_words = table['xml_tablewrap_raw_text'].split(" ")

    table_words_set = set(table_words)

    table_text = " ".join(table_words)
    candidate_page_nums = []
    scores = [0 for page in doc]
    for page_num, page in enumerate(doc):
        page_words_set = set([word[4] for word in page.get_text_words()])

        scores[page_num] = len(table_words_set.intersection(page_words_set))
    table_page = int(np.argmax(scores))

    return table_page, scores


def get_table_page_slow(doc, table, candidate_page_nums=None):
    table_words = []
    table_text = table['xml_tablewrap_raw_text']
    
    if not candidate_page_nums:
        candidate_page_nums = range(len(doc))
    scores = [0 for page_num in candidate_page_nums]
    for idx, page in enumerate(candidate_page_nums):
        page = doc[candidate_page_nums[idx]]
        page_words = page.get_text_words()
        sorted_words = sorted(page_words, key=functools.cmp_to_key(compare_meta))
        page_text = " ".join([word[4] for word in sorted_words])
        
        X = page_text.replace("~", "^")
        Y = table_text.replace("~", "^")

        score = align(X, Y, match_reward=2, mismatch_penalty=-2, new_gap_penalty=-10,
          continue_gap_penalty = -0.0005, page_boundary_gap_reward = 0.0001, score_only=True,
          gap_character='~')

        scores[idx] = score
        table_page = candidate_page_nums[int(np.argmax(scores))]

    return table_page, scores


def get_table_page(doc, table):
    table_page_num, scores = get_table_page_fast(doc, table)
    max_score = max(scores)
    candidate_page_nums = []
    for idx, score in enumerate(scores):
        if score >= max_score / 2:
            candidate_page_nums.append(idx)

    if len(candidate_page_nums) > 1:
        table_page_num, scores = get_table_page_slow(doc, table,
                                                     candidate_page_nums=candidate_page_nums)
        
    return table_page_num, scores


def locate_table(page, table):
    words = page.get_text_words()
    sorted_words = sorted(words, key=functools.cmp_to_key(compare_meta))
    page_text = " ".join([word[4] for word in sorted_words])

    page_text_source = []
    for num, word in enumerate(sorted_words):
        for c in word[4]:
            page_text_source.append(num)
        page_text_source.append(None)
    page_text_source = page_text_source[:-1]
        
    table_text = " ".join([entry['xml_text_content'].strip() for entry in table['cells']])
    table_text_source = []
    for num, cell in enumerate(table['cells']):
        for c in cell['xml_text_content'].strip():
            table_text_source.append(num)
        table_text_source.append(None)
    table_text_source = table_text_source[:-1]

    X = page_text.replace("~", "^")
    Y = table_text.replace("~", "^")

    match_reward = 3
    mismatch_penalty = -2
    new_gap_penalty = -10
    continue_gap_penalty = -0.05
    page_boundary_gap_reward = 0.2

    alignment, score = align(X, Y, match_reward=match_reward, mismatch_penalty=mismatch_penalty,
                             new_gap_penalty=new_gap_penalty, continue_gap_penalty=continue_gap_penalty,
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
                if word_num:
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


def locate_caption(page, caption):
    words = page.get_text_words()
    sorted_words = sorted(words, key=functools.cmp_to_key(compare_meta))
    page_text = " ".join([word[4] for word in sorted_words])

    page_text_source = []
    for num, word in enumerate(sorted_words):
        for c in word[4]:
            page_text_source.append(num)
        page_text_source.append(None)

    X = page_text.replace("~", "^")
    Y = caption.replace("~", "^")

    match_reward = 3
    mismatch_penalty = -2
    new_gap_penalty = -10
    continue_gap_penalty = -0.05
    page_boundary_gap_reward = 0.2

    alignment, score = align(X, Y, match_reward=match_reward, mismatch_penalty=mismatch_penalty,
                             new_gap_penalty=new_gap_penalty, continue_gap_penalty=continue_gap_penalty,
                             page_boundary_gap_reward=page_boundary_gap_reward, score_only=False,
          gap_character='~')
    
    matching_words = set()
    count = 0
    for char1, char2 in zip(alignment[0], alignment[1]):
        if not char1 == "~":
            if char1 == char2:
                matching_words.add(page_text_source[count])
            count += 1
            
    inliers = []
    for word_num in matching_words:
        if word_num:
            inliers.append(sorted_words[word_num])
    
    if len(inliers) == 0:
        return [], []
    
    bbox = list(inliers[0][0:4])
    for word in inliers[1:]:
        bbox[0] = min(bbox[0], word[0])
        bbox[1] = min(bbox[1], word[1])
        bbox[2] = max(bbox[2], word[2])
        bbox[3] = max(bbox[3], word[3])
    
    return bbox, inliers


def is_portrait(page, bbox):
    if bbox:
        bbox = fitz.Rect(bbox)
    else:
        bbox = page.rect
    portrait_count = 0
    landscape_count = 0
    page_dict = page.get_text("dict")
    for block in page_dict['blocks']:
        if 'lines' in block:
            for line in block['lines']:
                line_bbox = fitz.Rect(line['bbox'])
                if bbox and line_bbox in bbox:
                    direction = line['dir']
                    if direction[0] == 1 and direction[1] == 0:
                        portrait_count += 1
                    elif direction[0] == 0 and direction[1] == -1:
                        landscape_count += 1
    return portrait_count >= landscape_count
        
        
def save_full_tables_annotation(tables, document_annotation_filepath):
    # Remove "word_bboxes" field
    for table_dict in tables:
        if 'pdf_word_bboxes' in table_dict:
            del table_dict['pdf_word_bboxes']
        if 'pdf_caption_word_bboxes' in table_dict:
            del table_dict['pdf_caption_word_bboxes']
    
    with open(document_annotation_filepath, 'w', encoding='utf-8') as outfile:
        json.dump(tables, outfile, ensure_ascii=False, indent=4)


# Attempt to fix the caption and footer

# 1. Caption should encompass all of the "lines" that intersect the caption.
# 2. Footer should encompass all of the "blocks" that intersect the footer.

def fix_caption_and_footer(doc, table_dict):
    try:
        page = doc[table_dict['pdf_page_index']]
    except:
        return
    text = page.get_text('dict')
    blocks = text['blocks']
    block_bboxes = [block['bbox'] for block in blocks]

    try:
        caption_block_bboxes = []
        caption_bbox = table_dict['pdf_caption_bbox']
        for bbox in block_bboxes:
            if Rect(bbox).intersects(caption_bbox):
                caption_block_bboxes.append(bbox)
        caption_rect = Rect(caption_bbox)
        for bbox in caption_block_bboxes:
            caption_rect.include_rect(bbox)
        table_dict['pdf_caption_bbox'] = list(caption_rect)
    except:
        pass

    try:
        footer_block_bboxes = []
        footer_bbox = table_dict['pdf_table_footer_bbox']
        for bbox in block_bboxes:
            if Rect(bbox).intersects(footer_bbox):
                footer_block_bboxes.append(bbox)
        footer_rect = Rect(footer_bbox)
        for bbox in footer_block_bboxes:
            footer_rect.include_rect(bbox)
        table_dict['pdf_table_footer_bbox'] = list(footer_rect)
    except:
        pass
    
    try:
        table_wrap_rect = Rect(table_dict['pdf_table_wrap_bbox'])
        try:
            table_wrap_rect.include_rect(footer_rect)
        except:
            pass
        try:
            table_wrap_rect.include_rect(caption_rect)
        except:
            pass
        table_dict['pdf_table_wrap_bbox'] = list(table_wrap_rect)
    except:
        pass


def clean_xml_annotation(table_dict):
    num_columns = table_dict['num_columns']
    num_rows = table_dict['num_rows']
    header_rows = set()
    for cell in table_dict['cells']:
        if cell['is_column_header']:
            header_rows = header_rows.union(set(cell['row_nums']))
    num_header_rows = len(header_rows)
    
    #---REMOVE EMPTY ROWS---
    has_content_by_row = defaultdict(bool)
    for cell in table_dict['cells']:
        has_content = len(cell['xml_text_content'].strip()) > 0
        for row_num in cell['row_nums']:
            has_content_by_row[row_num] = has_content_by_row[row_num] or has_content
    table_dict['section_last_rows'] = [num_header_rows-1]
    row_count = num_header_rows
    for row_num in range(num_header_rows+1, num_rows):
        if not has_content_by_row[row_num]:
            table_dict['section_last_rows'].append(row_count)
        else:
            row_count += 1
    row_num_corrections = np.cumsum([int(not has_content_by_row[row_num]) for row_num in range(num_rows)]).tolist()
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
        table_dict['cells'].remove(cell)
    table_dict["num_rows"] = sum([int(elem) for idx, elem in has_content_by_row.items()])
        
    #---REMOVE EMPTY COLUMNS---
    has_content_by_column = defaultdict(bool)
    for cell in table_dict['cells']:
        has_content = len(cell['xml_text_content'].strip()) > 0
        for column_num in cell['column_nums']:
            has_content_by_column[column_num] = has_content_by_column[column_num] or has_content
    column_num_corrections = np.cumsum([int(not has_content_by_column[column_num]) for column_num in range(num_columns)]).tolist()
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
        table_dict['cells'].remove(cell)
    table_dict["num_columns"] = sum([int(elem) for idx, elem in has_content_by_column.items()])
    
    
def standardize_and_fix_xml_annotation(table_dict):
    num_columns = table_dict['num_columns']
    num_rows = table_dict['num_rows']
    
    #---IF FIRST ROW HAS CELL WITH COLSPAN > 1, MUST BE A HEADER
    first_row_has_colspan = False
    for cell in table_dict['cells']:
        if 0 in cell['row_nums'] and len(cell['column_nums']) > 1:
            first_row_has_colspan = True
    if first_row_has_colspan:
        for cell in table_dict['cells']:
            if 0 in cell['row_nums']:
                cell['is_column_header'] = True
    
    #---STANDARDIZE HEADERS: HEADERS END WITH A ROW WITH NO SUPERCELLS---
    cell_counts_by_row = defaultdict(int)
    header_status_by_row = defaultdict(bool)
    for cell in table_dict['cells']:
        for row_num in cell['row_nums']:
            if len(cell['xml_text_content'].strip()) == 0:
                cell_count = len(cell['column_nums'])
            else:
                cell_count = 1
            cell_counts_by_row[row_num] += cell_count
            if cell['is_column_header']:
                header_status_by_row[row_num] = True                
    true_header_status_by_row = defaultdict(bool)
    if header_status_by_row[0]:
        for row_num in range(num_rows):
            true_header_status_by_row[row_num] = True
            if cell_counts_by_row[row_num] == num_columns:
                break                
    true_header_rows = set([row_num for row_num, header_status in true_header_status_by_row.items() if header_status])
    for cell in table_dict['cells']:
        cell['is_column_header'] = len(set(cell['row_nums']).intersection(true_header_rows)) > 0
        
    #---STANDARDIZE HEADERS: IF FIRST COLUMN IN HEADER IS BLANK, HEADER CONTINUES UNTIL NON-BLANK CELL---
    min_nonblank_first_column_row = num_rows
    header_rows = set()
    for cell in table_dict['cells']:
        if cell['is_column_header']:
            for row_num in cell['row_nums']:
                header_rows.add(row_num)
        if 0 in cell['column_nums'] and len(cell['xml_text_content'].strip()) > 0:
            min_nonblank_first_column_row = min(min_nonblank_first_column_row, min(cell['row_nums']))
    if len(header_rows) > 0 and min_nonblank_first_column_row > max(header_rows) + 1:
        header_rows = set(range(min_nonblank_first_column_row))
    for cell in table_dict['cells']:
        if header_rows & set(cell['row_nums']):
            cell['is_column_header'] = True
        
    #---STANDARDIZE PROJECTED ROW HEADERS: ABSORB BLANK CELLS INTO NON-BLANK CELLS---
    non_projected_row_header_status_by_row = defaultdict(bool)
    first_cell_by_row = dict()
    cells_to_delete = []
    for cell in table_dict['cells']:
        # If there is a non-blank cell after the first column in the body, can't be a projected row header
        if (not cell['is_column_header'] and len(cell['xml_text_content'].strip()) > 0
            and min(cell['column_nums']) > 0 and len(cell['row_nums']) == 1):
            non_projected_row_header_status_by_row[cell['row_nums'][0]] = True
        # Note the first cell in each row, if it's not a supercell
        elif len(cell['xml_text_content'].strip()) > 0 and min(cell['column_nums']) == 0 and len(cell['row_nums']) == 1:
            first_cell_by_row[cell['row_nums'][0]] = cell
    for cell in table_dict['cells']:
        if (not cell['is_column_header'] and len(cell['xml_text_content'].strip()) == 0
            and min(cell['column_nums']) > 0 and len(cell['row_nums']) == 1):
            try:
                row_num = cell['row_nums'][0]
                if non_projected_row_header_status_by_row[row_num]:
                    continue
                cell_to_join_with = first_cell_by_row[row_num]
                cell_to_join_with['pdf_bbox'] = list(
                    Rect(cell_to_join_with['pdf_bbox']).include_rect(cell['pdf_bbox']))
                cell_to_join_with['column_nums'] = list(set(cell_to_join_with['column_nums'] + cell['column_nums']))
                cells_to_delete.append(cell)
            except:
                pass
    for cell in cells_to_delete:
        table_dict['cells'].remove(cell)
        
    #---LABEL PROJECTED ROW HEADERS---
    for cell in table_dict['cells']:
        if not cell['is_column_header'] and len(cell['column_nums']) == num_columns:
            cell['is_projected_row_header'] = True
        else:
            cell['is_projected_row_header'] = False
        
    #---STANDARDIZE SUPERCELLS IN FIRST COLUMN: ABSORB BLANK CELLS INTO NON-BLANK CELLS---
    first_column_cells_with_content_by_row = dict()
    # Determine cells with content
    for cell in table_dict['cells']:
        if 0 in cell['column_nums']:
            if len(cell['xml_text_content'].strip()) == 0:
                continue
            for row_num in cell['row_nums']:
                first_column_cells_with_content_by_row[row_num] = cell
    # For cells without content, determine cell to combine with
    cells_to_delete = []
    for cell in table_dict['cells']:
        if 0 in cell['column_nums']:
            if len(cell['xml_text_content'].strip()) == 0:
                cell_to_join_with = None
                for row_num in range(min(cell['row_nums'])-1, -1, -1):
                    if row_num in first_column_cells_with_content_by_row:
                        cell_to_join_with = first_column_cells_with_content_by_row[row_num]
                        break
                if not cell_to_join_with is None:
                    # Cells must have same header status and same column numbers to be joined
                    if not (set(cell_to_join_with['column_nums']) == set(cell['column_nums'])
                            and cell_to_join_with['is_column_header'] == cell['is_column_header']):
                        continue
                    cell_to_join_with['row_nums'] = list(set(cell_to_join_with['row_nums'] + cell['row_nums']))
                    try:
                        cell_to_join_with['pdf_bbox'] = list(
                            Rect(cell_to_join_with['pdf_bbox']).include_rect(cell['pdf_bbox']))
                    except:
                        pass
                    cells_to_delete.append(cell)                    
    for cell in cells_to_delete:
        table_dict['cells'].remove(cell)


def aggregate_cell_bboxes(page, table_dict, cell_bboxes, rotated=False):
    table_bbox = None
    row_bboxes = {}
    col_bboxes = {}
    expanded_cell_bboxes = {}
    cells = table_dict['cells']

    for cell_num, cell in enumerate(cells):
        try:
            cell_bbox = cell_bboxes[cell_num]
        except:
            continue
        if not cell_bbox:
            continue
            
        if not table_bbox:
            table_bbox = [entry for entry in cell_bbox]
        else:
            table_bbox = [min(table_bbox[0], cell_bbox[0]),
                          min(table_bbox[1], cell_bbox[1]),
                          max(table_bbox[2], cell_bbox[2]),
                          max(table_bbox[3], cell_bbox[3])]
            
    if table_bbox:
        if is_portrait(page, table_bbox):
            table_dict['pdf_is_rotated'] = 0
        else:
            table_dict['pdf_is_rotated'] = 1
        rotated = bool(table_dict['pdf_is_rotated'])
    
    for cell_num, cell in enumerate(cells):
        max_row = max(cell['row_nums'])
        min_row = min(cell['row_nums'])
        max_col = max(cell['column_nums'])
        min_col = min(cell['column_nums'])
        
        if not min_col in col_bboxes:
            col_bboxes[min_col] = [None, None, None, None]
        if not min_row in row_bboxes:
            row_bboxes[min_row] = [None, None, None, None]
        if not max_col in col_bboxes:
            col_bboxes[max_col] = [None, None, None, None]
        if not max_row in row_bboxes:
            row_bboxes[max_row] = [None, None, None, None]
        
        try:
            cell_bbox = cell_bboxes[cell_num]
        except:
            continue
            
        cell_bbox = cell_bboxes[cell_num]
        if not cell_bbox:
            continue
        
        if not rotated:
            if col_bboxes[min_col][0]:
                col_bboxes[min_col][0] = min(col_bboxes[min_col][0], cell_bbox[0])
            else:
                col_bboxes[min_col][0] = cell_bbox[0]

            if row_bboxes[min_row][1]:
                row_bboxes[min_row][1] = min(row_bboxes[min_row][1], cell_bbox[1])
            else:
                row_bboxes[min_row][1] = cell_bbox[1]

            if col_bboxes[max_col][2]:
                col_bboxes[max_col][2] = max(col_bboxes[max_col][2], cell_bbox[2])
            else:
                col_bboxes[max_col][2] = cell_bbox[2]

            if row_bboxes[max_row][3]:
                row_bboxes[max_row][3] = max(row_bboxes[max_row][3], cell_bbox[3])
            else:
                row_bboxes[max_row][3] = cell_bbox[3]
        else:
            if col_bboxes[min_col][1]:
                col_bboxes[min_col][1] = min(col_bboxes[min_col][1], cell_bbox[1])
            else:
                col_bboxes[min_col][1] = cell_bbox[1]

            if row_bboxes[min_row][0]:
                row_bboxes[min_row][0] = min(row_bboxes[min_row][0], cell_bbox[0])
            else:
                row_bboxes[min_row][0] = cell_bbox[0]

            if col_bboxes[max_col][3]:
                col_bboxes[max_col][3] = max(col_bboxes[max_col][3], cell_bbox[3])
            else:
                col_bboxes[max_col][3] = cell_bbox[3]

            if row_bboxes[max_row][2]:
                row_bboxes[max_row][2] = max(row_bboxes[max_row][2], cell_bbox[2])
            else:
                row_bboxes[max_row][2] = cell_bbox[2]
    if not rotated:
        for row_num in row_bboxes:
            row_bboxes[row_num][0] = table_bbox[0]
            row_bboxes[row_num][2] = table_bbox[2]

        for col_num in col_bboxes:
            col_bboxes[col_num][1] = table_bbox[1]
            col_bboxes[col_num][3] = table_bbox[3]
    else:
        for row_num in row_bboxes:
            row_bboxes[row_num][1] = table_bbox[1]
            row_bboxes[row_num][3] = table_bbox[3]

        for col_num in col_bboxes:
            col_bboxes[col_num][0] = table_bbox[0]
            col_bboxes[col_num][2] = table_bbox[2]
        
    for cell_num, cell in enumerate(cells):
        max_row = max(cell['row_nums'])
        min_row = min(cell['row_nums'])
        max_col = max(cell['column_nums'])
        min_col = min(cell['column_nums'])
        if not rotated:
            expanded_cell_bbox = [col_bboxes[min_col][0],
                                  row_bboxes[min_row][1],
                                  col_bboxes[max_col][2],
                                  row_bboxes[max_row][3]]
        else:
            expanded_cell_bbox = [row_bboxes[min_row][0],
                                  col_bboxes[min_col][1],
                                  row_bboxes[max_row][2],
                                  col_bboxes[max_col][3]]
        expanded_cell_bboxes[cell_num] = expanded_cell_bbox
    
    return table_bbox, col_bboxes, row_bboxes, expanded_cell_bboxes


def table_text_edit_distance(cells):
    L = 0
    D = 0
    for cell in cells:
        xml_text = ''.join(cell['xml_text_content'].split())
        pdf_text = ''.join(cell['pdf_text_content'].split())
        L += len(xml_text) + len(pdf_text)
        D += 2 * editdistance.eval(xml_text, pdf_text)
    try:
        D = D / L
    except:
        D = 1
            
    return D, L


def is_good_bbox(bbox, page_bbox):
    if (not bbox[0] is None and not bbox[1] is None and not bbox[2] is None and not bbox[3] is None
            and bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] <= page_bbox[2] and bbox[3] <= page_bbox[3]
            and bbox[0] < bbox[2]-1 and bbox[1] < bbox[3]-1):
        return True
    return False


def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1).intersect(bbox2)
    return intersection.get_area() / Rect(bbox1).get_area()


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
    
    ET.SubElement(bndbox, "xmin").text = "{0:.4f}".format(xmin)
    ET.SubElement(bndbox, "ymin").text = "{0:.4f}".format(ymin)
    ET.SubElement(bndbox, "xmax").text = "{0:.4f}".format(xmax)
    ET.SubElement(bndbox, "ymax").text = "{0:.4f}".format(ymax)
    
    return object_


def create_pascal_voc_object_element_direct(class_name, bbox):    
    object_ = ET.Element("object")
    name = ET.SubElement(object_, "name").text = class_name
    pose = ET.SubElement(object_, "pose").text = "Frontal"
    truncated = ET.SubElement(object_, "truncated").text = "0"
    difficult = ET.SubElement(object_, "difficult").text = "0"
    occluded = ET.SubElement(object_, "occluded").text = "0"
    bndbox = ET.SubElement(object_, "bndbox")
    
    ET.SubElement(bndbox, "xmin").text = "{0:.4f}".format(bbox[0])
    ET.SubElement(bndbox, "ymin").text = "{0:.4f}".format(bbox[1])
    ET.SubElement(bndbox, "xmax").text = "{0:.4f}".format(bbox[2])
    ET.SubElement(bndbox, "ymax").text = "{0:.4f}".format(bbox[3])
    
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


def save_json(page_annotation_json, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(page_annotation_json, outfile, indent=4)


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

def temp():
    count = 0
    good_count = 0
    annotation_files_by_pmc_id = [] # fix this
    total = len(annotation_files_by_pmc_id)
    for idx in range(0, total):
        pmc_id = pmc_ids[idx]
        annotation_file = annotation_files_by_pmc_id[pmc_id]
        count += 1
        if count < restart_point:
            continue

        annotation_filepath = os.path.join(annotations_directory, annotation_file)
            
        # Read table annotations, if they exist
        # Open table file
        with open(annotation_filepath, 'r', encoding='utf-8') as infile:
            table_dicts = json.load(infile)
            
        # Each table has associated bounding boxes
        for table_num, table_entry in enumerate(table_dicts):
            try:
                if not 'pdf' in table_entry:
                    continue
                split = table_entry['split']
                rotated = table_entry['pdf']['rotated']
                
                page_num = table_entry['pdf']['page_num']          
        
                # Create detection PASCAL VOC XML file and page image
                page_bbox = table_entry['pdf']['page']['bbox']

                # Get page image            
                table_image_filename = pmc_id + "_table_" + str(table_num) + ".jpg"
                table_image_filepath = os.path.join(structure_image_directory, table_image_filename)
                if os.path.exists(table_image_filepath):
                    img = Image.open(table_image_filepath)
                else:
                    print(table_image_filepath)
                    
                xml_filename = pmc_id + "_table_" + str(table_num) + ".xml"
                xml_filepath = os.path.join(structure_pascal_voc_directory, split, xml_filename)
                if not os.path.exists(xml_filepath):
                    print(xml_filepath)
                    
                page_words_filepath = os.path.join(page_words_data_directory, pmc_id + "_" + str(page_num) + ".json")
                with open(page_words_filepath, 'r') as f:
                    page_words = json.load(f)
                if 'words' in page_words:
                    page_words = page_words['words']
                    
                # Crop
                table_bbox = table_entry['pdf_bbox']
                crop_bbox = [table_bbox[0]-padding,
                            table_bbox[1]-padding,
                            table_bbox[2]+padding,
                            table_bbox[3]+padding]
                zoom = 1000 / max(page_bbox)
                
                tokens_in_table_img = get_tokens_in_table_img(page_words, crop_bbox)
                #print(tokens_in_table_img)
                
                for token in tokens_in_table_img:
                    bbox = token['bbox']
                    bbox[0] = bbox[0] * zoom
                    bbox[1] = bbox[1] * zoom
                    bbox[2] = bbox[2] * zoom
                    bbox[3] = bbox[3] * zoom
                    token['bbox'] = bbox
                
                # Convert to image coordinates
                crop_bbox = [int(round(zoom*elem)) for elem in crop_bbox]
                
                # Keep within image
                crop_bbox = [max(0, crop_bbox[0]),
                            max(0, crop_bbox[1]),
                            min(img.size[0], crop_bbox[2]),
                            min(img.size[1], crop_bbox[3])]
                
                for token in tokens_in_table_img:
                    bbox = token['bbox']
                    bbox[0] = (bbox[0] - crop_bbox[0] - 1)
                    bbox[1] = (bbox[1] - crop_bbox[1] - 1)
                    bbox[2] = (bbox[2] - crop_bbox[0] - 1)
                    bbox[3] = (bbox[3] - crop_bbox[1] - 1)
                    token['bbox'] = bbox
                
                #img = img.crop(crop_bbox)                    
                    
                # If rotated, rotate:
                if rotated:
                    for entry in tokens_in_table_img:
                        bbox = entry['bbox']
                        bbox = [img.size[0]-bbox[3]-2,bbox[0],img.size[0]-bbox[1]-2,bbox[2]]
                        entry['bbox'] = bbox
                        
                table_words_filename = pmc_id + "_table_" + str(table_num) + "_words.json"
                table_words_filepath = os.path.join(table_words_data_directory, table_words_filename)
                #print(table_words_filepath)
                with open(table_words_filepath, 'w', encoding='utf8') as f:
                    json.dump(tokens_in_table_img, f)

            except KeyboardInterrupt:
                break
            except FileNotFoundError:
                print("error")
                print(traceback.format_exc())
                pass
            except Exception as err:
                print("error")
                print(traceback.format_exc())
                print("idx: {}".format(idx))
            
        print("{}/{}".format(idx, total), end="\r")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        help="Root directory for source data to process")
    parser.add_argument('--output_dir',
                        help="Root directory for output data")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--timeout_seconds', type=int, default=90)
    parser.add_argument('--det_db_name', default='My-PubTables-Detection')
    parser.add_argument('--str_db_name', default='My-PubTables-Structure')
    parser.add_argument('--train_padding', type=int, default=30,
                        help="The amount of padding to add around a table in the training set when cropping.")
    parser.add_argument('--test_padding', type=int, default=5,
                        help="The amount of padding to add around a table in the val and test sets when cropping.")
    return parser.parse_args()


def main():
    args = get_args()

    source_directory = args.data_dir
    timeout_seconds = args.timeout_seconds
    detection_db_name = args.det_db_name
    structure_db_name = args.str_db_name
    VERBOSE = args.verbose
    # TODO: Incorporate a train/test/val split and change padding for test/val sets
    padding = args.train_padding

    output_directory = args.output_dir # root location where to save data
    det_xml_dir = os.path.join(output_directory, detection_db_name, "unsplit_xml")
    det_img_dir = os.path.join(output_directory, detection_db_name, "images")
    det_words_dir = os.path.join(output_directory, detection_db_name, "words")
    str_xml_dir = os.path.join(output_directory, structure_db_name, "unsplit_xml")
    str_img_dir = os.path.join(output_directory, structure_db_name, "images")
    str_words_dir = os.path.join(output_directory, structure_db_name, "words")
    pdf_annot_dir = os.path.join(output_directory, "My-PubTables-PDF-Annotations")

    dirs = [det_xml_dir, det_img_dir, det_words_dir,
            str_xml_dir, str_img_dir, str_words_dir, pdf_annot_dir]
    for di in dirs:
        if not os.path.exists(di):
            os.makedirs(di)

    '''
    Get a list of all PDF-XML document pairs
    '''
    xml_files_by_pmc_id = {}
    pdf_files_by_pmc_id = {}

    pdf_list_filepath = os.path.join(source_directory, "pdf_filelist.txt")
    if os.path.exists(pdf_list_filepath):
        with open(pdf_list_filepath) as f:
            for elem in f.readlines():
                elem = elem.strip()
                elem_xml = elem.replace(".pdf", ".nxml")
                pmc_id = elem.split("/")[0]
                xml_files_by_pmc_id[pmc_id] = elem_xml
                pdf_files_by_pmc_id[pmc_id] = elem
    else:
        dirs = os.listdir(source_directory)
        print(dirs)
        for d in dirs:
            if d.startswith("PMC"):
                dir_path = os.path.join(source_directory, d)
                if not os.path.isdir(dir_path):
                    continue
                dir_files = os.listdir(dir_path)
                for fi in dir_files:
                    if fi.endswith(".pdf"):
                        if fi.replace(".pdf", ".nxml") in dir_files:
                            pdf_file = os.path.join(d, fi)
                            pdf_files_by_pmc_id[d] = pdf_file
                            xml_files_by_pmc_id[d] = pdf_file.replace(".pdf", ".nxml")

    pdf_files = list(pdf_files_by_pmc_id.values())

    num_files = len(pdf_files)
    print(num_files)

    output_image_max_dim = 1000

    start_point = 0
    end_point = 10 #start_point + 2

    # Progress tracking
    skipped_files = 0
    parsed_instances = 0
    annotated_tables = 0
    exception_count = 0
    timeout_count = 0
    table_encounters = 0
    low_quality_tables = 0
    tables_for_detection = 0
    table_image_count = 0

    '''
    Process each PDF-XML file pair to annotate the tables in the PDF.
    '''
    print("Num files total: {}".format(num_files))
    print("Annotating files {}-{}".format(start_point, end_point-1))
    for idx, pdf_file in enumerate(pdf_files):
        if idx < start_point:
            continue
        if idx >= end_point:
            break
        
        pmc_id = pdf_file.split("/")[0]
        split = "train" # fix this: split_by_pmc_id[pmc_id]
        save_filepath = os.path.join(pdf_annot_dir, pmc_id + "_tables.json")
        
        print(pdf_file)
        pdf_path = os.path.join(source_directory, pdf_file)
        xml_path = pdf_path.replace(".pdf", ".nxml")
        
        print("PROGRESS: {}/{} ({}/{}). {} files skipped. Table instances: {} parsed, {} fully annotated.  ".format(
            idx-start_point+1, end_point-start_point, idx, num_files, skipped_files, parsed_instances, 
            annotated_tables), flush=True)
        
        try:       
            if VERBOSE: print("Reading XML...", end='')
            xml_string = read_xml(xml_path)
            #print(xml_string)
            table_dicts = extract_table_xmls_from_document(xml_string)
            
            num_tables_parsed = len(table_dicts)
            parsed_instances += num_tables_parsed
            if VERBOSE: print(" {} tables found.".format(num_tables_parsed))
        except Exception as e:
            print(e)
            print("Error during XML reading/parsing.")
            skipped_files += 1
            continue
            
        if num_tables_parsed == 0:
            continue

        try:
            print("Reading PDF")
            doc = read_pdf(pdf_path)
        except:
            print("Error during PDF reading.")
            skipped_files += 1
            continue

        annotated_table_dicts = []
        for table_idx, table_dict in enumerate(table_dicts):
            table_dict['xml_table_index'] = table_idx
            
            table_encounters += 1
            try:
                #-------------------#
                # PARSING
                #-------------------#
                if VERBOSE: print("Parsing table XML...")
                start_index = table_dict['xml_table_wrap_start_character_index']
                end_index = table_dict['xml_table_wrap_end_character_index']
                table_xml = xml_string[start_index:end_index]
                #print(table_xml)
                table_dict = parse_xml_table(xml_string, table_dict)
                #print(table_dict)

                if table_dict:
                    if len(table_dict['cells']) == 0:
                        print("Table has no annotated cells; could be a graphic")
                        continue
                    clean_xml_annotation(table_dict)
                    
                    #-------------------#
                    # ALIGNMENT
                    #-------------------#                
                    if VERBOSE: print("Locating table page...")
                    with timeout(seconds=timeout_seconds):
                        table_page_num, scores = get_table_page(doc, table_dict)
                        table_dict['pdf_page_index'] = table_page_num
                    print(table_page_num)

                    page = doc[table_page_num]
                    page_words = get_page_words(page)
                    table_dict['pdf_full_page_bbox'] = list(page.rect)

                    if VERBOSE: print("Locating table bounding box...")
                    with timeout(seconds=timeout_seconds):
                        cell_bboxes, word_bboxes = locate_table(page, table_dict)

                    if not cell_bboxes:
                        print("No cell bboxes")
                        continue

                    #-------------------#
                    # COMPLETION
                    #-------------------#
                    table_bbox, col_bboxes, row_bboxes, expanded_cell_bboxes = aggregate_cell_bboxes(page, table_dict,
                                                                                cell_bboxes)
                    if VERBOSE:
                        print("Table frame bbox: {}".format(table_bbox))
                    table_dict['pdf_table_bbox'] = table_bbox

                    caption_pieces = []
                    try:
                        label = table_dict['xml_caption_label_text']
                        if len(label) > 0:
                            caption_pieces.append(label)
                    except:
                        print(traceback.format_exc())

                    try:
                        text = table_dict['xml_caption_text']
                        if len(text) > 0:
                            caption_pieces.append(text)
                    except:
                        print(traceback.format_exc())
                    caption = " ".join(caption_pieces)

                    caption_bbox = []
                    table_dict['pdf_caption_page_num'] = None
                    table_dict['pdf_caption_bbox'] = []
                    table_dict['pdf_caption_word_bboxes'] = []
                    if len(caption) > 9:
                        if VERBOSE: print("Locating caption bounding box...")
                        with timeout(seconds=60):
                            caption_bbox, caption_word_bboxes = locate_caption(page, caption)

                        if len(caption_bbox) == 4:
                            table_dict['pdf_caption_page_num'] = table_page_num
                            table_dict['pdf_caption_bbox'] = caption_bbox
                            table_dict['pdf_caption_word_bboxes'] = caption_word_bboxes
                    if VERBOSE: print("Caption bbox: {}".format(caption_bbox))

                    footer_bbox = []
                    footer = table_dict['xml_table_footer_text']
                    table_dict['pdf_table_footer_page_num'] = None
                    table_dict['pdf_table_footer_bbox'] = []
                    table_dict['pdf_table_footer_word_bboxes'] = []
                    if len(footer) > 9:
                        if VERBOSE: print("Locating footer bounding box...")
                        with timeout(seconds=60):
                            footer_bbox, footer_word_bboxes = locate_caption(page, footer)

                        if len(footer_bbox) == 4:
                            table_dict['pdf_table_footer_page_num'] = table_page_num
                            table_dict['pdf_table_footer_bbox'] = footer_bbox
                            table_dict['pdf_table_footer_word_bboxes'] = footer_word_bboxes
                    if VERBOSE: print("Footer bbox: {}".format(footer_bbox))
                        
                    header_rows = set()
                    for cell in table_dict['cells']:
                        if cell['is_column_header']:
                            header_rows = header_rows.union(set(cell['row_nums']))

                    table_wrap_bbox = Rect(table_bbox)
                    if len(caption_bbox) == 4:
                        table_wrap_bbox.include_rect(caption_bbox)
                    if len(footer_bbox) == 4:
                        table_wrap_bbox.include_rect(footer_bbox)
                    table_dict['pdf_table_wrap_bbox'] = list(table_wrap_bbox)
                    table_dict['rows'] = [{} for row_num in row_bboxes]
                    for row_num in row_bboxes:
                        table_dict['rows'][row_num]['pdf_row_bbox'] = row_bboxes[row_num]
                        table_dict['rows'][row_num]['is_column_header'] = row_num in header_rows
                    table_dict['columns'] = [{} for col_num in col_bboxes]
                    for column_num in col_bboxes:
                        table_dict['columns'][column_num]['pdf_column_bbox'] = col_bboxes[column_num]
                    for cell_num, cell in enumerate(table_dict['cells']):
                        cell['pdf_bbox'] = expanded_cell_bboxes[cell_num]
                        cell['pdf_text_tight_bbox'] = cell_bboxes[cell_num]
                        if not cell_bboxes[cell_num] is None:
                            pdf_text_content, _ = extract_text_inside_bbox(page_words, cell_bboxes[cell_num])
                        else:
                            pdf_text_content = ""
                        cell['pdf_text_content'] = pdf_text_content
                    table_dict['pdf_word_bboxes'] = word_bboxes

                    #-------------------#
                    # CANONICALIZATION
                    #-------------------#
                    try:
                        fix_caption_and_footer(doc, table_dict)
                        
                        if VERBOSE:
                            print("Adjusted caption bbox: {}".format(table_dict['pdf_caption_bbox']))
                            print("Adjusted footer bbox: {}".format(table_dict['pdf_table_footer_bbox']))
                    except:
                        print(traceback.format_exc())
                    try:
                        standardize_and_fix_xml_annotation(table_dict)
                    except:
                        print(traceback.format_exc())
                        
                    #-------------------#
                    # QUALITY CONTROL
                    #-------------------#
                    D, _ = table_text_edit_distance(table_dict['cells'])
                    if VERBOSE:
                        print("Table text content annotation disagreement score: {}".format(D))
                    if D > 0.05:
                        low_quality_tables += 1
                        print(">>>> LOW QUALITY TABLE ANNOTATION <<<<")
                        continue
                        
                    #-------------------#
                    # CLEANUP
                    #-------------------#
                    if 'pdf_word_bboxes' in table_dict:
                        del table_dict['pdf_word_bboxes']
                    if 'pdf_caption_word_bboxes' in table_dict:
                        del table_dict['pdf_caption_word_bboxes']
                    if 'pdf_table_footer_word_bboxes' in table_dict:
                        del table_dict['pdf_table_footer_word_bboxes']
                        
                    #-------------------#
                    # SAVE
                    #-------------------#
                        
                    table_dict['pmc_id'] = pmc_id
                    table_dict['pdf_file_name'] = pdf_file.split("/")[-1]
                    table_dict['xml_file_name'] = table_dict['pdf_file_name'].replace(".pdf", ".nxml")
                    table_dict['split'] = split
                    table_dict['structure_id'] = pmc_id + "_table_" + str(table_idx)

                    annotated_tables += 1
                    annotated_table_dicts.append(table_dict)
            except KeyboardInterrupt:
                break
            except TimeoutError:
                table_dict['timeout'] = True
                print("#### TIMEOUT ####")
                annotated_table_dicts.append(table_dict)
                timeout_count += 1
            except:
                print("**** EXCEPTION ****")
                print(traceback.format_exc())
                exception_count += 1
                
        # While this is pretty strict and will exclude some tables for detection that could be included,
        # it's the only way to ensure that tables included for detection are correct using an automated method.
        # If every table in the XML passes the checks, then we know that no page image will have a missing or
        # incomplete table annotation.
        if len(annotated_table_dicts) == len(table_dicts):
            exclude_for_detection = False
            tables_for_detection += len(table_dicts)
        else:
            exclude_for_detection = True
            if VERBOSE:
                print("Not all tables fully annotated and passed quality control. Exclude these for detection.")
        for table_dict in annotated_table_dicts:
            table_dict['exclude_for_detection'] = exclude_for_detection
            
        #-----------------------------------#
        # TABLE DETECTION IMAGE DATA
        #-----------------------------------#
        if not exclude_for_detection:
            detection_boxes_by_page = defaultdict(list)     
            
            # Each table has associated bounding boxes
            for table_dict in annotated_table_dicts:
                if 'timeout' in table_dict and table_dict['timeout']:
                    continue
                try:
                    table_boxes = []

                    rotated = table_dict['pdf_is_rotated']
                    page_num = table_dict['pdf_page_index']

                    # Create detection data
                    if rotated:
                        class_label = 'table rotated'
                    else:
                        class_label = 'table'
                    dict_entry = {'class_label': class_label, 'bbox': table_dict['pdf_table_bbox']}
                    detection_boxes_by_page[page_num].append(dict_entry)
                except Exception as err:
                    print(traceback.format_exc())

                # Create detection PASCAL VOC XML file and page image
                for page_num, boxes in detection_boxes_by_page.items():
                    try:
                        page_bbox = list(doc[page_num].rect)
                        if not all([is_good_bbox(entry['bbox'], page_bbox) for entry in boxes]):
                            raise Exception("At least one bounding box has non-positive area or is outside of image")

                        # Create page image            
                        image_filename = pmc_id + "_" + str(page_num) + ".jpg"
                        image_filepath = os.path.join(det_img_dir, image_filename)
                        img = create_document_page_image(doc, page_num, output_image_max_dim=output_image_max_dim)
                        img.save(image_filepath)

                        # Initialize PASCAL VOC XML
                        page_annotation = create_pascal_voc_page_element(image_filename, img.width, img.height,
                                                                        database=detection_db_name)

                        for entry in boxes:
                            # Add to PASCAl VOC
                            element = create_pascal_voc_object_element(entry['class_label'],
                                                                    entry['bbox'], page_bbox,
                                                                    output_image_max_dim=output_image_max_dim)
                            page_annotation.append(element)   
                            
                        xml_filename = pmc_id + "_" + str(page_num) + ".xml"
                        xml_filepath = os.path.join(det_xml_dir, xml_filename)
                        save_xml_pascal_voc(page_annotation, xml_filepath)

                        page_words_filename = xml_filename.replace(".xml", "_words.json")
                        page_words_filepath = os.path.join(det_words_dir, page_words_filename)

                        page = doc[page_num]
                        page_words = get_page_words(page)
                        zoom = 1000 / max(page.rect)
                        for word in page_words:
                            word['bbox'] = [zoom*elem for elem in word['bbox']]
                        with open(page_words_filepath, 'w') as f:
                            json.dump(page_words, f)
                    except Exception as err:
                        print(traceback.format_exc())

        # Save results for this document
        if VERBOSE: print(save_filepath)
        save_full_tables_annotation(annotated_table_dicts, save_filepath)
        if VERBOSE: print("DETECTION SAMPLE SAVED!")

        #-----------------------------------#
        # TABLE STRUCTURE IMAGE DATA
        #-----------------------------------#
        for table_num, table_entry in enumerate(annotated_table_dicts):
            if 'timeout' in table_entry and table_entry['timeout']:
                continue

            try:
                table_boxes = []
                    
                # Check if table has at least two columns
                num_columns = table_entry['num_columns']
                if num_columns < 2:
                    continue
                if not 'columns' in table_entry:
                    continue
                    
                # Check if table has at least one row
                num_rows = table_entry['num_rows']
                if num_rows < 1:
                    continue
                if not 'rows' in table_entry:
                    continue
                
                rotated = table_entry['pdf_is_rotated']
                
                page_num = table_entry['pdf_page_index']          

                # Create structure recognition data
                dict_entry = {'class_label': 'table', 'bbox': table_entry['pdf_table_bbox']}
                table_boxes.append(dict_entry)
                
                # Dilation
                if rotated:
                    row1_idx = 2
                    row2_idx = 0
                    col1_idx = 1
                    col2_idx = 3
                else:
                    row1_idx = 3
                    row2_idx = 1
                    col1_idx = 2
                    col2_idx = 0
                rows = table_entry['rows']
                rows = sorted(rows, key=lambda k: k['pdf_row_bbox'][row1_idx]) 
                if len(rows) > 1:
                    for row1, row2 in zip(rows[:-1], rows[1:]):
                        mid_point = (row1['pdf_row_bbox'][row1_idx] + row2['pdf_row_bbox'][row2_idx]) / 2
                        row1['pdf_row_bbox'][row1_idx] = mid_point
                        row2['pdf_row_bbox'][row2_idx] = mid_point
                columns = table_entry['columns']
                if rotated:
                    columns = sorted(columns, key=lambda k: -k['pdf_column_bbox'][col1_idx])
                else:
                    columns = sorted(columns, key=lambda k: k['pdf_column_bbox'][col1_idx]) 
                if len(columns) > 1:
                    for col1, col2 in zip(columns[:-1], columns[1:]):
                        mid_point = (col1['pdf_column_bbox'][col1_idx] + col2['pdf_column_bbox'][col2_idx]) / 2
                        col1['pdf_column_bbox'][col1_idx] = mid_point
                        col2['pdf_column_bbox'][col2_idx] = mid_point
                for cell in table_entry['cells']:
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
                for cell in table_entry['cells']:
                    cell_bbox = cell['pdf_bbox']
                    blank = len(cell['xml_text_content'].strip()) == 0
                    supercell = len(cell['row_nums']) > 1 or len(cell['column_nums']) > 1
                    header = cell['is_column_header']
                    if not header and len(cell['column_nums']) == num_columns:
                        dict_entry = {'class_label': 'table projected row header', 'bbox': cell['pdf_bbox']}
                        table_boxes.append(dict_entry)                      
                    elif supercell and not blank:
                        dict_entry = {'class_label': 'table spanning cell', 'bbox': cell['pdf_bbox']}
                        table_boxes.append(dict_entry)                     
                        
                    if header:
                        header_rect.include_rect(cell_bbox)

                if header_rect.get_area() > 0:
                    dict_entry = {'class_label': 'table column header', 'bbox': list(header_rect)}
                    table_boxes.append(dict_entry)
                        
                for row in table_entry['rows']:
                    row_bbox = row['pdf_row_bbox']
                    dict_entry = {'class_label': 'table row', 'bbox': row_bbox}
                    table_boxes.append(dict_entry) 
                
                # table_entry['columns']
                for column in table_entry['columns']:
                    dict_entry = {'class_label': 'table column', 'bbox': column['pdf_column_bbox']}
                    table_boxes.append(dict_entry) 
        
                # Create detection PASCAL VOC XML file and page image
                page_bbox = table_entry['pdf_full_page_bbox']

                # Get page image            
                page_image_filename = pmc_id + "_" + str(page_num) + ".jpg"
                page_image_filepath = os.path.join(det_img_dir, page_image_filename)
                if os.path.exists(page_image_filepath):
                    img = Image.open(page_image_filepath)
                else:
                    #print("Creating image")
                    img = create_document_page_image(doc, page_num, output_image_max_dim=1000)
                    
                # Crop
                table_bbox = table_entry['pdf_table_bbox']
                crop_bbox = [table_bbox[0]-padding,
                            table_bbox[1]-padding,
                            table_bbox[2]+padding,
                            table_bbox[3]+padding]
                zoom = 1000 / max(page_bbox)
                
                # Convert to image coordinates
                crop_bbox = [int(round(zoom*elem)) for elem in crop_bbox]
                
                # Keep within image
                crop_bbox = [max(0, crop_bbox[0]),
                            max(0, crop_bbox[1]),
                            min(img.size[0], crop_bbox[2]),
                            min(img.size[1], crop_bbox[3])]
                
                img = img.crop(crop_bbox)                    
                for entry in table_boxes:
                    bbox = entry['bbox']
                    bbox = [zoom*elem for elem in bbox]
                    bbox = [bbox[0]-crop_bbox[0]-1,
                            bbox[1]-crop_bbox[1]-1,
                            bbox[2]-crop_bbox[0]-1,
                            bbox[3]-crop_bbox[1]-1]
                    entry['bbox'] = bbox
                    
                # If rotated, rotate:
                if rotated:
                    img = img.rotate(270, expand=True)
                    for entry in table_boxes:
                        bbox = entry['bbox']
                        bbox = [img.size[0]-bbox[3]-1,bbox[0],img.size[0]-bbox[1]-1,bbox[2]]
                        entry['bbox'] = bbox
                
                # Initialize PASCAL VOC XML
                table_image_filename = pmc_id + "_table_" + str(table_num) + ".jpg"
                table_image_filepath = os.path.join(str_img_dir, table_image_filename)
                table_annotation = create_pascal_voc_page_element(table_image_filename,
                                                                 img.width, img.height,
                                                                 database=structure_db_name)

                for entry in table_boxes:
                    bbox = entry['bbox']

                    # Add to PASCAl VOC
                    element = create_pascal_voc_object_element_direct(entry['class_label'],
                                                                      entry['bbox'])
                    table_annotation.append(element)              

                img.save(table_image_filepath)

                xml_filename = pmc_id + "_table_" + str(table_num) + ".xml"
                xml_filepath = os.path.join(str_xml_dir, xml_filename)
                if VERBOSE: print(xml_filepath)
                save_xml_pascal_voc(table_annotation, xml_filepath)

                table_words_filename = xml_filename.replace(".xml", "_words.json")
                table_words_filepath = os.path.join(str_words_dir, table_words_filename)

                page_words = get_page_words(doc[page_num])
                table_words = []
                for word_num, word in enumerate(page_words):
                    token = {}
                    token['flags'] = 0
                    token['span_num'] = word_num
                    token['line_num'] = 0
                    token['block_num'] = 0
                    bbox = [round(zoom * v, 5) for v in word['bbox']]
                    if iob(bbox, crop_bbox) > 0.75:
                        bbox = [max(0, bbox[0]-crop_bbox[0]-1),
                                max(0, bbox[1]-crop_bbox[1]-1),
                                min(img.size[0], bbox[2]-crop_bbox[0]-1),
                                min(img.size[1], bbox[3]-crop_bbox[1]-1)]
                        if (bbox[0] < 0 or bbox[1] < 0 or bbox[2] > img.size[0] or bbox[3] > img.size[1]
                            or bbox[0] > bbox[2] or bbox[1] > bbox[3]):
                            bad_box = True
                        else:
                            token['bbox'] = bbox
                            token['text'] = word['text']
                            table_words.append(token)
                with open(table_words_filepath, 'w') as f:
                    json.dump(table_words, f)

                table_image_count += 1
                if VERBOSE: print("STRUCTURE SAMPLE SAVED!")
            except KeyboardInterrupt:
                break
            except Exception as err:
                print("error")
                print(traceback.format_exc())
                print("idx: {}".format(idx))

    print("Number of table encounters: {}".format(table_encounters))
    print("Number of table annotations completed: {}".format(annotated_tables))
    print("Number of tables eligible for detection: {}".format(tables_for_detection))
    print("Number of low quality tables removed: {}".format(low_quality_tables))
    print("Number of exceptions: {}".format(exception_count))
    print("Numer of timeouts: {}".format(timeout_count))
    print("Numer of cropped tables saved in PASCAL VOC format: {}".format(table_image_count))

if __name__ == "__main__":
    main()