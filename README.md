# Table Transformer

This repository contains training and evaluation code for table extraction. We
perform the experiments on PubTables1M. The code adopts DETR and provides
supporting training code. In addition to the training code, we also provide the
code to evaluate GriTS metrics proposed in the paper. Please download the
[PubTables1M](https://pubtables1m.blob.core.windows.net/pubtables1m/README)
dataset before going forward.

## Installation
Create a conda environment from the yml file and activate it as follows

> conda env create -f environment.yml
> conda activate tables-detr

## Training
There's 2 tasks in this dataset.

1. Table Detection: Detecting tables in pdf page images
2. Table Structure Recognition: Detection of table structures

For Detailed description, please refer to the paper.

Sample training commands look like the below:

> cd src
> python main.py --data_root_dir /path/to/detection --data_type detection
> python main.py --data_root_dir /path/to/structure --data_type structure

## GriTS metric evaluation
GriTS metrics proposed in the paper can be evaluated once you have trained a
model. We consider the model trained in the previous step. This script
calculates all 4 variations presented in the paper. Based on the model, one can
tune which variation to use. The table words dir path is not required for all
variations but we use it in our case as PubTables1M contains this information.

> python grits.py --data_root_dir /path/to/structure --model_load_path /path/to/model --table_words_dir /path/to/table/words


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
