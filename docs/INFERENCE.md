# Inference

With the Table Transformer (TATR) inference pipeline you can:
1. *Detect* all tables in a document image.
2. *Recognize* the structure of a table in a cropped table image and output to HTML or CSV (and other formats).
3. *Extract* (detect and recognize) all tables in a document image in a single step.

Recognized/extracted tables can be output in multiple formats: 1. raw object bounding boxes; 2. list of cells, with properties; 3. HTML; 4. CSV.

Of these, the list of cells (#2) contains the most information, as it includes the bounding box of every cell. The HTML table format is somewhat lossy because it does not include bounding box information for the cells. The CSV format is slightly more lossy in that the first row and only the first row contains the column headers/names. So multi-row column headers must be "flattened" into a single row.

## Sample Code
Converting a cropped table image to HTML or CSV:
```
from inference import TableExtractionPipeline

# Create inference pipeline
pipe = TableExtractionPipeline(det_config_path='detection_config.json', det_model_path='../pubtables1m_detection_detr_r18.pth', det_device='cuda', str_config_path='structure_config.json', str_model_path='../pubtables1m_structure_detr_r18.pth', str_device='cuda')

# Recognize table(s) from image
extracted_tables = pipe.recognize(img, tokens, out_objects=True, out_cells=True, out_html=True, out_csv=True)

# Select table (there could be more than one)
extracted_table = extracted_tables[0]

# Get output in desired format
objects = extracted_table['objects']
cells = extracted_table['cells']
csv = extracted_table['csv']
html = extracted_table['html']
```

## Model Files
To run the pipelines, you need to provide config and model checkpoint files.

For table detection you need:
1. A detection model config JSON file
2. A pre-trained detection model checkpoint file

For table structure recognition you need:
1. A structure model config JSON file
2. A pre-trained structure model checkpoint file

For end-to-end table extraction you need all four of the above files.


## Pipeline Input Formats
When running the sample code:
- `img` is expected to be of type `PIL.Image`.
- `tokens` is expected to be a list of dictionaries

`tokens` contains a list of words and their bounding boxes in image coordinates. It is assumed to be sorted in reading order. The format for `tokens` is:
```
[
    {
        'bbox': [0.0, 0.0, 50.0, 50.0]
        'text': 'First'
    },
    {
        'bbox': [52.0, 0.0, 102.0, 50.0]
        'text': 'next'
    }
]
```
where `bbox` is in `[xmin, ymin, xmax, ymax]` format.
## Running From the Command Line
Change to `src` directory:
```
cd src
```
### To run table detection on a folder of document page images:
```
python inference.py --mode detect --detection_config_path detection_config.json --detection_model_path ../pubtables1m_detection_detr_r18.pth --detection_device cuda --image_dir [PATH TO DOCUMENT PAGE IMAGES] --words_dir [OPTIONAL PATH TO WORDS (ex. OCR) EXTRACTED FROM DOCUMENT PAGE IMAGES] --out_dir [PATH TO SAVE DETECTION OUTPUT] [FLAGS: -o,-z,-v,-p] --crop_padding 20
```
where:
- `--words_dir` is optional. But text content cannot be included in the final result without it.
- `--crop_padding` determines how much padding in pixels will be added around a detected table before outputting a cropped image of the table. This amount should be adjusted based on the structure recognition model that will be used on these images.
- `-o` means to output detected objects (with bounding boxes).
- `-p` means to save cropped images of the detected tables and save a JSON with cropped words/tokens (if they were supplied as input).
- `-v` means to print (verbose) output to the console.
- `-z` means to create and save (visualize) figures depicting the detected tables.

### To run table structure recognition on a folder of cropped table images:
```
python inference.py --mode recognize --structure_config_path structure_config.json --structure_model_path ../pubtables1m_structure_detr_r18.pth --structure_device cuda --image_dir [PATH TO CROPPED TABLE IMAGES] --words_dir [OPTIONAL PATH TO WORDS (ex. OCR) EXTRACTED FROM CROPPED TABLE IMAGES] --out_dir [PATH TO SAVE DETECTION OUTPUT] [FLAGS: -o,-c,-l,-m,-z,-v]
 ```
 where:
 - `--words_dir` is optional. But text content cannot be included in the final result without it. You can still use the visualize flag (-z) to see a depiction of the recognized table structure even without supplying text as input.
- `-o` means to output detected objects (with bounding boxes)
- `-l` means to output a list of recognized cells and their properties
- `-m` means to output the recognized table in HTML format
- `-c` means to output the recognized table in CSV format
- `-v` means to print (verbose) output to the console
- `-z` means to create and save (visualize) figures depicting the recognized tables and recognized cells in the tables

 ### To run table extraction (detection and recognition combined end-to-end) on a folder of document page images:
```
python inference.py --mode extract --detection_config_path detection_config.json --detection_model_path ../pubtables1m_detection_detr_r18.pth --detection_device cuda --structure_config_path structure_config.json --structure_model_path ../pubtables1m_structure_detr_r18.pth --structure_device cuda --image_dir [PATH TO DOCUMENT PAGE IMAGES] --words_dir [OPTIONAL PATH TO WORDS (ex. OCR) EXTRACTED FROM DOCUMENT PAGE IMAGES] --out_dir [PATH TO SAVE DETECTION OUTPUT] [FLAGS: -o,-c,-l,-m,-z,-v,-p] --crop_padding 20
```
where:
 - `--words_dir` is optional. But text content cannot be included in the final result without it. You can still use the visualize flag (-z) to see a depiction of the recognized table structure even without supplying text as input.
