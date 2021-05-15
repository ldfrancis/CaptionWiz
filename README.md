# CaptionWiz
An implementation of image captioning algorithms.

# Installation
Follow the following steps to install captionwiz
1. Clone the repository. 
```bash
git clone repository
```

2. It is advised to use miniconda and create a conda virtual environment. Having created a virtual environment proceed to the next step

2. Install requirements
```bash
cd CaptionWiz
pip install -e .
pip install -r requirements/requirements.txt
```

# Usage
To use captionwiz, you specify the dataset and the caption model to use in the config file - config.yaml. Analyzing the dataset, training and testing/inference can be done with the following commands.

**For dataset analysis**:
```bash
captionwiz --analyze_data
```
or
```bash
captionwiz -ad
```

**For training**:
```bash
captionwiz --train
```

**For testing**:
```bash
captionwiz --test
```
