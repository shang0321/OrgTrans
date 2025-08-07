# OrgTrans: A Transfer Learning Framework for Domain-generalized Morphological Analysis of Organoids

**Authors:**  

<img src="https://github.com/shang0321/OrgTrans/raw/master/assets/OrgTrans1.png" alt="OrgTrans1 Image" width="500" height="300"/>


## 📝Introduction

OrgTrans, a cross-domain organoid analysis pipeline, harnesses transfer learning and domain-adaptive modules to enable robust detection and classification across heterogeneous imaging data. It provides scalable and generalizable support for multi-source organoid analysis.

![Workflow](./worker.png)

## 🧪Visualizations

| ![1](file:///D:/_system/Download/1.png) | ![2](file:///D:/_system/Download/2.png) |
| --------------------------------------- | --------------------------------------- |
| ![3](file:///D:/_system/Download/3.png) | ![4](file:///D:/_system/Download/4.png) |

## 📦Getting Started

<img src="https://img.shields.io/badge/python-3.10.x%20|%203.11.x-blueviolet" alt="Python 3.10.x | 3.11.x">

---

### • Environment Setup Configuration

Automatic environment setup, please be patient.

```bash

# This creates an isolated environment to avoid conflicts with existing packages
conda create -n orgline_env python=3.10

# This switches to the OrgLine environment for package installation
conda activate orgtrans_env

# Git is required to clone the repository from GitHub
conda install git

# This downloads the complete source code to your local machine
git clone https://github.com/shang0321/OrgTrans.git

# Change to the project directory containing all necessary files
cd OrgTrans

# Install dependencies from requirements.txt
pip install -r requirements.txt
```



## ➡️Usage

---
### • Dataset Preparation

- Properly organize your organoid datasets for training and inference by following this directory structure:

```bash
OrgTrans/
├── datasets/
│   ├── image/                # Training set
│   │   ├── train/           # Training images
│   │   ├── test/            # Test images
│   │   └── val/             # Validation images
│   ├── label/                # Bounding box annotations
│   │   ├── train/           # Annotations for training images
│   │   ├── test/            # Annotations for test images (optional)
│   │   └── val/             # Annotations for validation images
│   ├── create_txt.py         # Script to create .txt files for dataset
│   └── generated_files/      # Folder to store generated files
│       ├── test.txt          # Contains paths of the test set images
│       ├── train_1_percent.txt # Contains paths of 1% of the training set (fully-supervised)
│       ├── unlabeled_1_percent.txt # Contains paths of 1% of the training set (unlabeled)
│       └── val.txt           # Contains paths of the validation set images
│
└── # After running create_txt.py, the following files will be generated:
    ├── test.txt              # Contains the test set image paths
    ├── train_1_percent.txt   # Contains 1% of the training set images (fully-supervised)
    ├── unlabeled_1_percent.txt # Contains 1% of the training set images (unlabeled)
    └── val.txt               # Contains the validation set image paths
```

- Execute the following command to generate data set splits:

```bash
# This switches to the OrgTrans environment for package installation
conda activate orgtrans_env

# Navigate to the datasets directory
cd datasets

# You can modify the instructions to choose any proportion of fully-supervised labels, such as 1%, 5%, or any other value. Here’s how you can update it:
# Make sure that 'data' and 'create_txt.py' are in the same directory.
python create_txt.py
```



### • Training

```shell script
# Navigate to the supervised training configuration directory
cd OrgTrans/configs/sup
# Run supervised training using the 'sup.yaml' configuration file
python train.py --cfg configs/sup/sup.yaml

# Navigate to the semi-supervised training configuration directory
cd OrgTrans/configs/ssod
# Run semi-supervised training using the 'transfer_ssod.yaml' configuration file and load the weights from supervised training
python train.py --cfg configs/ssod/transfer_ssod.yaml
```

### • Testing the Semi-Supervised Model

```shell script
python val.py --cfg configs/ssod/transfer_ssod.yaml --weights 
```



## ⚖️License

OrgLine is released under the [MIT License](MIT-License.txt), a permissive open-source license that allows for free use, modification, distribution, and private use of the software. This license requires that the original copyright notice and permission notice be included in all copies or substantial portions of the software.



## 👍Acknowledgement

We would like to thank the authors of [efficientteacher](https://github.com/AlibabaResearch/efficientteacher).



## 📃Citation

If you use this codebase in your research or project, please cite:

