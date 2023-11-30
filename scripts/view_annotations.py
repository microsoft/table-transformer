"""
Copyright (C) 2023 Microsoft Corporation

Assumes the data is in PASCAL VOC data format and the folder structure is:
[data_directory]/
- images/
- train/
- test/
- val/
"""

import argparse
import os
# import json
from collections import defaultdict
import traceback

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import xml.etree.ElementTree as ET

def read_pascal_voc(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes = []
    labels = []

    for object_ in root.iter('object'):

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
    # parser.add_argument('--words_data_dir',
    #                     help="Root directory for source data to process")
    parser.add_argument('--split', default='',
                        help="Split to process")
    parser.add_argument('--output_dir',
                        help="Root directory for output data")
    parser.add_argument('--num_samples', type=int)
    return parser.parse_args()

def main():
    args = get_args()

    data_directory = args.pascal_data_dir
    # words_directory = args.words_data_dir
    split = args.split
    output_directory = args.output_dir
    num_samples = args.num_samples

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    xml_filenames = [elem for elem in os.listdir(os.path.join(data_directory, split)) if elem.endswith(".xml")]

    for idx, filename in enumerate(xml_filenames):
        if not num_samples is None and idx == num_samples:
            break
        print(filename)
        try:
            xml_filepath = os.path.join(data_directory, split, filename)
            img_filepath = xml_filepath.replace(split, "images").replace(".xml", ".jpg")
            
            bboxes, labels = read_pascal_voc(xml_filepath)
            img = Image.open(img_filepath)

            # TODO: Add option to include words
            #words_filepath = os.path.join(words_directory, filename.replace(".xml", "_words.json"))
            #try:
            #    with open(words_filepath, 'r') as json_file:
            #        words = json.load(json_file)
            #except:
            #    words = []
            
            ax = plt.gca()
            ax.imshow(img, interpolation="lanczos")

            _, a, b = min(((abs(a * img.size[0] - b * img.size[1]), a, b) for a in range(1, 7) for b in range(1, 7) if 6 <= a * b < 6 + min(a, b)))

            _, axs = plt.subplots(b, a, figsize=(12, 12))
            axes = axs.flatten()
            for ax in axes:
                ax.imshow(img, interpolation="lanczos")

            # plt.gcf().set_size_inches((24 * 3, 24 * 2))

            tables = [bbox for bbox, label in zip(bboxes, labels) if label == 'table']
            columns = [bbox for bbox, label in zip(bboxes, labels) if label == 'table column']
            rows = [bbox for bbox, label in zip(bboxes, labels) if label == 'table row']
            column_headers = [bbox for bbox, label in zip(bboxes, labels) if label == 'table column header']
            projected_row_headers = [bbox for bbox, label in zip(bboxes, labels) if label == 'table projected row header']
            spanning_cells = [bbox for bbox, label in zip(bboxes, labels) if label == 'table spanning cell']

            ax = axes[0]
            for table_bbox in tables:
                rect = patches.Rectangle(table_bbox[:2], table_bbox[2]-table_bbox[0], table_bbox[3]-table_bbox[1], linewidth=1, 
                                         edgecolor='green', facecolor="none"
                                         # , alpha=0.25
                                         )
                ax.add_patch(rect)

            column_even_edge_color = 'darkorange'
            column_even_face_color = 'orange'
            column_even_alpha = 0.1
            column_odd_edge_color = 'darkblue'
            column_odd_face_color = 'blue'
            column_odd_alpha = 0.2
            ax = axes[1]
            for column_num, bbox in enumerate(columns):
                if column_num % 2 == 0:
                    alpha = column_even_alpha
                    facecolor = column_even_face_color
                    edgecolor = column_even_edge_color
                else:
                    alpha = column_odd_alpha
                    facecolor = column_odd_face_color
                    edgecolor = column_odd_edge_color
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0, 
                                         edgecolor=edgecolor, facecolor=facecolor, linestyle="-",
                                         alpha=alpha)
                ax.add_patch(rect)
                # rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                #                              edgecolor='rosybrown', facecolor='none', linestyle="-",
                #                              alpha=0.8)
                # ax.add_patch(rect)

            row_even_edge_color = 'darkred'
            row_even_face_color = 'red'
            row_even_alpha = 0.1
            row_odd_edge_color = 'darkgreen'
            row_odd_face_color = 'green'
            row_odd_alpha = 0.2
            ax = axes[2]
            for row_num, bbox in enumerate(rows):
                if row_num % 2 == 1:
                    alpha = row_odd_alpha
                    edgecolor = row_odd_edge_color
                    facecolor = row_odd_face_color
                else:
                    alpha = row_even_alpha
                    facecolor = row_even_edge_color
                    edgecolor = row_even_edge_color
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0, 
                                         edgecolor=edgecolor, facecolor=facecolor, linestyle="-",
                                         alpha=alpha)
                ax.add_patch(rect)
                # rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                #                              edgecolor='blue', facecolor='none', linestyle="-",
                #                              alpha=0.8)
                # ax.add_patch(rect)
            
            ax = axes[3]
            for bbox in column_headers:
                linewidth = 3
                alpha = 0.3
                facecolor = (1, 0, 0.75) #(0.5, 0.45, 0.25)
                edgecolor = (1, 0, 0.75) #(1, 0.9, 0.5)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                         edgecolor='none',facecolor=facecolor, alpha=alpha)
                ax.add_patch(rect)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0, 
                                         edgecolor=edgecolor,facecolor='none',linestyle="-", hatch='///')
                ax.add_patch(rect)

            ax = axes[4]
            for bbox in projected_row_headers:
                facecolor = (1, 0.9, 0.5) #(0, 0.75, 1) #(0, 0.4, 0.4)
                edgecolor = (1, 0.9, 0.5) #(0, 0.7, 0.95)
                alpha = 0.35
                linewidth = 3
                linestyle="--"
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                         edgecolor='none',facecolor=facecolor, alpha=alpha)
                ax.add_patch(rect)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1,
                                         edgecolor=edgecolor,facecolor='none',linestyle=linestyle)
                ax.add_patch(rect)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0,
                                         edgecolor=edgecolor,facecolor='none'
                                         #, linestyle=linestyle
                                         , hatch='\\\\'
                                         )
                ax.add_patch(rect)

            ax = axes[5]
            for bbox in spanning_cells:
                color = (0.2, 0.5, 0.2) #(0, 0.4, 0.4)
                alpha = 0.4
                linewidth = 4
                linestyle="-"
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                         edgecolor='none',facecolor=color, alpha=alpha)
                ax.add_patch(rect)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                         edgecolor=color,facecolor='none',linestyle=linestyle) # hatch='//'
                ax.add_patch(rect)

            for ax in axes:
              ax.set_xlim([min((table_bbox[0] for table_bbox in tables)) - 5, max((table_bbox[2] for table_bbox in tables))+5])
              ax.set_ylim([max((table_bbox[3] for table_bbox in tables))+5, min((table_bbox[1] for table_bbox in tables)) - 5])

            # table_bbox = tables[0]
            # plt.xlim([table_bbox[0]-5, table_bbox[2]+5])
            # plt.ylim([table_bbox[3]+5, table_bbox[1]-5])
            plt.xticks([], [])
            plt.yticks([], [])

            legend_elements = [[Patch(facecolor=(1, 1, 1), edgecolor="green",
                                     label='Tables')],
                                   [Patch(facecolor=column_odd_face_color, edgecolor=column_odd_edge_color, alpha=column_odd_alpha,
                                     label='Column (odd)'),
                               Patch(facecolor=column_even_face_color, edgecolor=column_even_edge_color, alpha=column_even_alpha,
                                     label='Column (even)')],
                           [Patch(facecolor=row_odd_face_color, edgecolor=row_odd_edge_color, alpha=row_odd_alpha,
                                     label='Row (odd)'),
                               Patch(facecolor=row_even_face_color, edgecolor=row_even_edge_color, alpha=row_even_alpha,
                                     label='Row (even)')],
                               [Patch(facecolor=(1, 0.7, 0.925), edgecolor=(1, 0, 0.75),
                                     label='Column header', hatch='///')],
                               [Patch(facecolor=(1, 0.965, 0.825), edgecolor=(1, 0.9, 0.5),
                                     label='Projected row header', hatch='\\\\')],
                               [Patch(facecolor=(0.68, 0.8, 0.68), edgecolor=(0.2, 0.5, 0.2),
                                     label='Spanning cell')],
                                     ]
            for ax, legend_elements in zip(axes, legend_elements):
               ax.legend(handles=legend_elements, bbox_to_anchor=(0, -0.02), loc='upper left', borderaxespad=0,
                         fontsize=16, ncol=4)
            plt.gcf().set_size_inches(20 * a * img.size[0] / b / img.size[1], 20)
            plt.axis('off')
            save_filepath = os.path.join(output_directory, filename.replace(".xml", "_ANNOTATIONS.jpg"))
            plt.savefig(save_filepath, bbox_inches='tight', dpi=150)
            # plt.show()
            plt.close()
        except:
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()