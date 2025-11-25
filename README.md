# HistoOmicsAlign
Bridging Genotype and Phenotype for Cancer Diagnosis and Survival Prediction by Integrative Analysis of Histopathology and Genomic Features

## Installation
First, locally install HistoOmicsAlign using pip.
Environments:
- Python 3.8
- PyTorch 2.4.0

```bash 
git clone https://github.com/jinqy-lzu/HistoOmicsAlign.git
cd HistoOmicsAlign
conda create --name HistoOmicsAlign python=3.8
conda activate HistoOmicsAlign
pip3 install -r requirements.txt
```

## Downloading TCGA Data and Genomics 
To download diagnostic WSIs (formatted as .svs files), molecular feature data and other clinical metadata, please refer  to the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov)and the [cBioPortal](https://www.cbioportal.org/). WSIs for each cancer type can be downloaded using the [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/). The mutation status, copy number variations, and RNA-seq data can be retrieved from the [cBioPortal for Cancer Genomics](https://www.cbioportal.org). 

## Processing Whole Slide Images 
To process Whole Slide Images (WSIs), first, the tissue regions in each biopsy slide are segmented using Otsu's Segmentation on a downsampled WSI using OpenSlide. The 224 x 224 patches without spatial overlapping are extracted from the segmented tissue regions at the desired magnification. Consequently, A Vision-Language Foundation Model for Computational Pathology [CONCH](https://github.com/mahmoodlab/CONCH) is used to encode raw image patches into feature vectors, which we then save as .pt files for each WSI. The extracted features then serve as input (in a .pt file) to the network.

## Code Base Structure
The code base structure is explained below: 
- **main.py**: Cross-validation script for training multimodal networks. This script will save evaluation metrics and predictions on the train + test split for each epoch on every split in **checkpoints**.
- **feature_fusion.py**: Contains PyTorch model definitions for fusion.
- **dataset.py**: Contains the PyTorch DatasetLoader definition for loading multimodal data.
- **optim.py**: Build optimizer.
- **model_ema.py**: Script for Exponential Moving Average Model(EMA). The EMA is a technique used for smoothing time series data. It reduces noise and volatility by weighted averaging the data, thereby extracting trends from the data. In deep learning, EMA is often used in the process of updating and optimizing model parameters. It can help the model converge more stably during the training process and improve its generalization ability.
- **misc.py**: Contains scripts for the checkpoint manager

The directory structure for your multimodal dataset should look similar to the following:
```bash
./
├── data
      └── PROJECT
            ├── pt_file (e.g. pt)
                ├── sample_001.pt
                ├── sample_002.t
                ├── ...
            ├── genomic_data.csv
├── config
      ├── Brain-LGG_surv.yaml
      ├── Kidney-KIRC_surv.ymal
      ├── ...
      ├── Brain_cls.ymal
      ├── Kidney_cls.ymal
      ├── ...
```
## Training and Evaluation
Here are example commands for training  multimodal networks.

### Survival Model
Example shown below for training a survival model and saving the model checkpoints + predictions at the end. We use the YAML file of the specified tumor for model training. In this example, we will train a survival risk prediction model on Low Grade Glioma tumor(LGG) data samples.

```
python main.py --config ./config/Brain-LGG_surv.ymal
```
### Classification Model
Example shown below for training a classification model and saving the model checkpoints + predictions at the end. We use the YAML file of the specified tumor for model training. In this example, we use brain tumor data to train a classification model that can distinguish different subtypes(Low-Grade Glioma (LGG) and Glioblastoma Multiforme (GBM)).

```
python main.py --config ./config/Brain_cls.ymal
```
