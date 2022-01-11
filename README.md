# PubTables-1M

This repository contains code and links to data for the papers:
- ["PubTables-1M: Towards comprehensive table extraction from unstructured documents"](https://arxiv.org/pdf/2110.00061.pdf)
- "GriTS: Grid table similarity metric for table structure recognition" (coming soon)

*Note: Updates to the code and papers (and documentation) are currently ongoing and we will announce when each of these is ready for a stable release.*

The goal of PubTables-1M is to create a large, detailed, high-quality dataset for training and evaluating a wide variety of models for the tasks of **table detection**, **table structure recognition**, and **functional analysis**.

![table_extraction_v2](https://user-images.githubusercontent.com/10793386/139559159-cd23c972-8731-48ed-91df-f3f27e9f4d79.jpg)

It contains:
- 460,589 annotated document pages containing tables for table detection.
- 947,642 fully annotated tables including text content and complete location (bounding box) information for table structure recognition and functional analysis.
- Full bounding boxes in both image and PDF coordinates for all table rows, columns, and cells (including blank cells), as well as other annotated structures such as column headers and projected row headers.
- Rendered images of all tables and pages.
- Bounding boxes and text for all words appearing in each table and page image.
- Additional cell properties not used in the current model training.

Additionally, cells in the headers are *canonicalized* and we implement multiple *quality control* steps to ensure the annotations are as free of noise as possible. For more details, please see [our paper](https://arxiv.org/pdf/2110.00061.pdf).

## News
`11/21/2021`: Our paper "PubTables-1M: Towards comprehensive table extraction from unstructured documents" is available on [arXiv](https://arxiv.org/pdf/2110.00061.pdf).\
`10/21/2021`: The full PubTables-1M dataset has been officially released on [Microsoft Research Open Data](https://msropendata.com/datasets/505fcbe3-1383-42b1-913a-f651b8b712d3).

## Getting the Data
[PubTables-1M](https://msropendata.com/datasets/505fcbe3-1383-42b1-913a-f651b8b712d3) is available for download from [Microsoft Research Open Data](https://msropendata.com/).

It comes in 5 tar.gz files:
- PubTables-1M-Image_Page_Detection_PASCAL_VOC.tar.gz
- PubTables-1M-Image_Page_Words_JSON.tar.gz
- PubTables-1M-Image_Table_Structure_PASCAL_VOC.tar.gz
- PubTables-1M-Image_Table_Words_JSON.tar.gz
- PubTables-1M-PDF_Annotations_JSON.tar.gz

To download from the command line:
1. Visit the [dataset home page](https://msropendata.com/datasets/505fcbe3-1383-42b1-913a-f651b8b712d3) with a web browser and click Download in the top left corner. This will create a link to download the dataset from Azure with a unique access token for you that looks like `https://msropendataset01.blob.core.windows.net/pubtables1m?[SAS_TOKEN_HERE]`.
2. You can then use the command line tool [azcopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) to download all of the files with the following command:
```
azcopy copy "https://msropendataset01.blob.core.windows.net/pubtables1m?[SAS_TOKEN_HERE]" "/path/to/your/download/folder/" --recursive
```

Then unzip each of the archives from the command line using:
```
tar -xzvf yourfile.tar.gz
```

## For Docker users

```bash
docker pull phamquiluan/table-transformer:latest
# or
docker build -t phamquiluan/table-transformer -f Dockerfile .

# train TSR
docker run -it --shm-size 8G --gpus all \
  -v <data-path>:/code/data \
  -v <output-path>:/code/output \
  -v phamquiluan/table-transformer \
  python3 main.py --data_root_dir /code/data --data_type structure
```


## Code Installation
Create a conda environment from the yml file and activate it as follows
```
conda env create -f environment.yml
conda activate tables-detr
```

## Model Training
The code trains models for 2 different sets of table extraction tasks:

1. Table Detection
2. Table Structure Recognition + Functional Analysis

For a detailed description of these tasks and the models, please refer to the paper.

Sample training commands:

```
cd src
python main.py --data_root_dir /path/to/detection --data_type detection
python main.py --data_root_dir /path/to/structure --data_type structure
```

## GriTS metric evaluation
GriTS metrics proposed in the paper can be evaluated once you have trained a
model. We consider the model trained in the previous step. This script
calculates all 4 variations presented in the paper. Based on the model, one can
tune which variation to use. The table words dir path is not required for all
variations but we use it in our case as PubTables1M contains this information.

```
python main.py --data_root_dir /path/to/structure --model_load_path /path/to/model --table_words_dir /path/to/table/words --mode grits
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
