import torch
from xml.dom import minidom
import xml.etree.ElementTree as ET
from fitz import Rect
from PIL import Image


def create_pascal_voc_page_element(
    image_pure_path, output_image_width, output_image_height, database
):
    if not image_pure_path.is_absolute():
        print("Warning: {} is not absolute.".format(image_pure_path))
    # Create XML of tables on PDF page in PASCAL VOC format
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder").text = image_pure_path.parent.name
    filename = ET.SubElement(annotation, "filename").text = image_pure_path.name
    path = ET.SubElement(annotation, "path").text = str(image_pure_path)
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


def read_pascal_voc(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes = []
    labels = []

    for object_ in root.iter("object"):
        ymin, xmin, ymax, xmax = None, None, None, None

        label = object_.find("name").text

        for box in object_.findall("bndbox"):
            ymin = float(box.find("ymin").text)
            xmin = float(box.find("xmin").text)
            ymax = float(box.find("ymax").text)
            xmax = float(box.find("xmax").text)

        bbox = [xmin, ymin, xmax, ymax]  # PASCAL VOC

        bboxes.append(bbox)
        labels.append(label)

    return bboxes, labels


def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1.tolist()).intersect(bbox2.tolist())
    return intersection.get_area() / Rect(bbox1.tolist()).get_area()


def optimal_width_height(image_size, n):
    return min(
        (
            (
                abs(a * image_size[0] - b * image_size[1]),
                a,
                b,
            )
            for a in range(n + 1)
            for b in range(n + 1)
            if n <= a * b < n + min(a, b)
        )
    )


def pad_torch_box(box, padding):
    return box + torch.tensor(
        [
            -padding,
            -padding,
            padding,
            padding,
        ]
    )


def rotate90_box_with_original_width(box, w):
    assert torch.all(box[..., 0] <= box[..., 2])
    assert torch.all(box[..., 1] <= box[..., 3])
    # box is one box or an tensor of boxes
    a = box.roll(1, dims=-1)
    a[..., 0] = w - a[..., 0]
    a[..., 2] = w - a[..., 2]
    assert torch.all(a[..., 0] <= a[..., 2])
    assert torch.all(a[..., 1] <= a[..., 3])
    return a


# def rotate270_box_with_original_width(box, w):
#     a = box.clone()
#     a[0] = w - a[0]
#     a[2] = w - a[2]
#     # box is one box or an tensor of boxes
#     return a.roll(-1)


def rotate270_boxes_with_original_width(box, w):
    assert torch.all(box[..., 0] <= box[..., 2])
    assert torch.all(box[..., 1] <= box[..., 3])
    a = box.clone()
    a[..., 0] = w - a[..., 0]
    a[..., 2] = w - a[..., 2]
    # box is either one box or a tensor of boxes
    a = a.roll(-1, dims=-1)
    assert torch.all(a[..., 0] <= a[..., 2])
    assert torch.all(a[..., 1] <= a[..., 3])
    return a

# def rotate90_boxes_with_original_width(boxes, w):
#     # box is one box or an tensor of boxes
#     a = boxes.roll(-1, dims=-1)
#     a[:, 0] = w - a[:, 0]
#     a[:, 2] = w - a[:, 2]
#     return a


def image_exists_and_is_valid(image_path):
    try:
        with Image.open(image_path) as im:
            im.verify()
    except:
        return False
    return True
