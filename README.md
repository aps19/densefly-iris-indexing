# Fast Similarity Search In large scale iris recognition

This repository contains the implementation of the "Fast Similarity Search in Large Scale Iris Recognition".

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.9.0 or higher
- facenet-pytorch
- OpenCV
- scikit-learn

You can install the prerequisites using pip:

```bash
pip install torch torchvision facenet-pytorch opencv-python scikit-learn
```
Clone the repo:
```bash
git clone https://github.com/aps19/densefly-iris-indexing.git
```
Running the scripts

Prepare Dataset
```bash
python prepare_data.py
```

Finetune Feature Extractor
```bash
python finetuning_model.py
```

DenseFly Hashing
```bash
python indexing.py
```
