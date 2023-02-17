"""
Copyright (C) 2023 Microsoft Corporation

Script to create a version of the dataset with a specified amount of padding around the table.
Does not add padding to the image, only crops the image to have the specified amount of padding
around the table.

Assumes the data is in PASCAL VOC data format and the folder structure is:
[data_directory]/
- images/
- train/
- test/
- val/
"""

import argparse
import os
from xml.dom import minidom
import xml.etree.ElementTree as ET
import json

from PIL import Image
from fitz import Rect

def read_pascal_voc(xml_filepath):

    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    bboxes = []
    labels = []
    
    filename = root.find('filename').text
    size = root.find('size')
    source = root.find('source')
    database = source.find('database').text
    width = size.find('width').text
    height = size.find('height').text

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

    return bboxes, labels, filename, width, height, database


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

def create_pascal_voc_object_element(class_label, bbox):
    object_ = ET.Element("object")
    name = ET.SubElement(object_, "name").text = class_label
    pose = ET.SubElement(object_, "pose").text = "Frontal"
    truncated = ET.SubElement(object_, "truncated").text = "0"
    difficult = ET.SubElement(object_, "difficult").text = "0"
    occluded = ET.SubElement(object_, "occluded").text = "0"
    bndbox = ET.SubElement(object_, "bndbox")
    
    ET.SubElement(bndbox, "xmin").text = str(bbox[0])
    ET.SubElement(bndbox, "ymin").text = str(bbox[1])
    ET.SubElement(bndbox, "xmax").text = str(bbox[2])
    ET.SubElement(bndbox, "ymax").text = str(bbox[3])
    
    return object_

def save_xml_pascal_voc(page_annotation, filepath):
    xmlstr = minidom.parseString(ET.tostring(page_annotation)).toprettyxml(indent="   ")
    with open(filepath, "w") as f:
        f.write(xmlstr)

def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1).intersect(bbox2)
    return intersection.get_area() / Rect(bbox1).get_area()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pascal_data_dir',
                        help="Root directory for source data to process")
    parser.add_argument('--words_data_dir',
                        help="Root directory for source data to process")
    parser.add_argument('--split', default='',
                        help="Split to process")
    parser.add_argument('--table_padding', type=int, default=2)
    return parser.parse_args()

def main():
    args = get_args()

    data_directory = args.pascal_data_dir
    if data_directory.endswith(os.sep):
        data_directory = data_directory[:-1]
    words_directory = args.words_data_dir
    if words_directory.endswith(os.sep):
        words_directory = words_directory[:-1]
    split = args.split
    padding = args.table_padding

    data_output_directory = data_directory + "_PADDING_" + str(padding)
    words_output_directory = words_directory + "_PADDING_" + str(padding)

    if not os.path.exists(data_output_directory):
        os.makedirs(data_output_directory)

    if not os.path.exists(words_output_directory):
        os.makedirs(words_output_directory)

    source_subdir = os.path.join(data_directory, split)
    dest_subdir = os.path.join(data_output_directory, split)
    source_image_directory = os.path.join(data_directory, "images")
    dest_image_directory = os.path.join(data_output_directory, "images")

    if not os.path.exists(dest_subdir):
        os.makedirs(dest_subdir)

    if not os.path.exists(dest_image_directory):
        os.makedirs(dest_image_directory)

    files = os.listdir(source_subdir)

    for file in files:
        filepath = os.path.join(source_subdir, file)
        words_filepath = os.path.join(words_directory, file.replace('.xml', '_words.json'))

        bboxes, labels, filename, width, height, database = read_pascal_voc(filepath)

        tables = [idx for idx, label in enumerate(labels) if label == 'table']

        image_filepath = os.path.join(source_image_directory, filename)

        img = Image.open(image_filepath)

        if not len(tables) == 1:
            print('Problem')

        table_bbox = bboxes[tables[0]]

        crop_bbox = [round(elem) for elem in table_bbox]
        crop_bbox[0] -= padding
        crop_bbox[1] -= padding
        crop_bbox[2] += padding
        crop_bbox[3] += padding

        img = img.crop(crop_bbox)

        annotation = create_pascal_voc_page_element(filename, img.width, img.height, database)

        for label, bbox in zip(labels, bboxes):
            bbox = [bbox[0]-crop_bbox[0],
                    bbox[1]-crop_bbox[1],
                    bbox[2]-crop_bbox[0],
                    bbox[3]-crop_bbox[1]]

            # Add to PASCAl VOC
            element = create_pascal_voc_object_element(label, bbox)
            annotation.append(element)  

        dest_img_path = os.path.join(dest_image_directory, filename)
        img.save(dest_img_path)
        dest_annot_path = os.path.join(dest_subdir, file)
        save_xml_pascal_voc(annotation, dest_annot_path)

        with open(words_filepath, 'r') as jf:
            data = json.load(jf)

        padded_words = []
        for word in data:
            bbox = word['bbox']

            if iob(bbox, crop_bbox) >= 0.5:

                bbox = [bbox[0]-crop_bbox[0],
                        bbox[1]-crop_bbox[1],
                        bbox[2]-crop_bbox[0],
                        bbox[3]-crop_bbox[1]]
                word['bbox'] = bbox
                padded_words.append(word)

        padded_words_filepath = os.path.join(words_output_directory,
                                             file.replace('.xml', '_words.json'))
        with open(padded_words_filepath, 'w') as jf:
            json.dump(padded_words, jf)

if __name__ == "__main__":
    main()