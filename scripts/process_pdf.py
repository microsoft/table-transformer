import argparse

import pytesseract
import numpy
import json

from PIL import Image
from pytesseract import Output
from pdf2image import convert_from_path
import cv2


class ImageRenderer:
    def draw(self, img, tokens):
        pass

    def show(self):
        pass


class OpenCvRenderer(ImageRenderer):
    def draw(self, img, tokens):
        for token in tokens:
            (x, y, w, h) = (token['bbox'][0], token['bbox'][1], token['bbox'][2], token['bbox'][3])
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
        cv2.imshow('img', img)

    def show(self):
        cv2.waitKey(0)


def load_input(path: str) -> list[Image]:
    if path.endswith(".pdf"):
        return convert_from_path(path)
    return [Image.open(path)]


def change_path_suffix(path: str, suffix: str) -> str:
    return "".join([path.rsplit(".", 1)[0], suffix])


def save(path, img, tokens):
    img.save(path)
    words_save_filepath = change_path_suffix(path, "_words.json")
    with open(words_save_filepath, 'w', encoding='utf8') as f:
        json.dump(tokens, f)


def process_pdf(src_file: str, renderer: ImageRenderer):
    images = load_input(src_file)
    for page in range(len(images)):
        img = numpy.asarray(images[page])
        d = pytesseract.image_to_data(img, output_type=Output.DICT)

        n_boxes = len(d['level'])
        tokens = []
        for i in range(n_boxes):
            token = {
                'flags': 0,
                'span_num': d['word_num'][i],
                'line_num': d['line_num'][i],
                'block_num': d['block_num'][i],
                'bbox': [d['left'][i], d['top'][i], d['left'][i] + d['width'][i], d['top'][i] + d['height'][i]],
                'text': str(d['text'][i]).strip()
            }
            if token['text'] == '':
                continue
            tokens.append(token)

        renderer.draw(img, tokens)

        save_filepath = change_path_suffix(src_file, "_" + str(page) + ".jpg")
        save(save_filepath, images[page], tokens)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_file', help="Pdf file to scan")
    return parser.parse_args()


def main():
    args = get_args()

    src_file = args.pdf_file

    renderer = ImageRenderer()
    process_pdf(src_file, renderer)
    renderer.show()


if __name__ == "__main__":
    main()
