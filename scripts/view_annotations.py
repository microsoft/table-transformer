import argparse
import os
import json
from collections import defaultdict, Counter
import traceback
from difflib import SequenceMatcher
import statistics
import random

import fitz
from fitz import Rect
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pdf2image
from pdf2image import convert_from_path
from PyPDF2 import PdfFileReader
import editdistance
import numpy as np

def read_pascal_voc(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes = []
    labels = []

    for object_ in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None
        
        label = object_.find("name").text

        for box in object_.findall("bndbox"):
            ymin = float(box.find("ymin").text)
            xmin = float(box.find("xmin").text)
            ymax = float(box.find("ymax").text)
            xmax = float(box.find("xmax").text)

        bbox = [xmin, ymin, xmax, ymax] # PASCAL VOC
        
        bboxes.append(bbox)
        labels.append(label)

    return bboxes, labels

color_map = defaultdict(lambda: ('magenta', 0, 1))
color_map.update({'table': ('brown', 0.1, 3), 'table row': ('blue', 0.04, 1),
                  'table column': ('red', 0.04, 1), 'table projected row header': ('cyan', 0.2, 3),
                  'table column header': ('magenta', 0.2, 3), 'table spanning cell': ('green', 0.6, 3)})

def plot_bbox(ax, bbox, color='magenta', linewidth=1, alpha=0):
    rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                             edgecolor='none',facecolor=color, alpha=alpha)
    ax.add_patch(rect)
    rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                             edgecolor=color,facecolor='none',linestyle="--")
    ax.add_patch(rect) 


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pascal_data_dir',
                        help="Root directory for source data to process")
    parser.add_argument('--words_data_dir',
                        help="Root directory for source data to process")
    parser.add_argument('--split',
                        help="Split to process")
    parser.add_argument('--output_dir',
                        help="Root directory for output data")
    return parser.parse_args()

def main():
    args = get_args()

    data_directory = args.pascal_data_dir
    words_directory = args.words_data_dir
    split = args.split
    output_directory = args.output_dir

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    xml_filenames = os.listdir(os.path.join(data_directory, split))

    for filename in xml_filenames:
        print(filename)
        xml_filepath = os.path.join(data_directory, split, filename)
        img_filepath = xml_filepath.replace(split, "images").replace(".xml", ".jpg")
        words_filepath = os.path.join(words_directory, filename.replace(".xml", "_words.json"))
        
        bboxes, labels = read_pascal_voc(xml_filepath)
        img = Image.open(img_filepath)
        with open(words_filepath, 'r') as json_file:
            words = json.load(json_file)
        
        ax = plt.gca()
        ax.imshow(img)
        for word in words:
            plot_bbox(ax, word['bbox'], color="orange", linewidth=0.5, alpha=0.1)
        for bbox, label in zip(bboxes, labels):
            color, alpha, linewidth = color_map[label]
            plot_bbox(ax, bbox, color=color, linewidth=linewidth, alpha=alpha)
        fig = plt.gcf()
        fig.set_size_inches((18, 18))
        plt.axis('off')
        plt.savefig('test.jpg', bbox_inches='tight', dpi=100)
        plt.show()
        plt.close()

if __name__ == "__main__":
    main()