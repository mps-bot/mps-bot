# MPS-Bot

The source code of MPS-Bot (Message Passing over Simplexes for Bot Detection)

## Environment Setup

### 1. Create and activate a conda environment

```bash
   conda create --name mpsbot python=3.7
   conda activate mpsbot
```

### 2. Install required dependencies:

```bash
   wget https://download.pytorch.org/whl/cu102/torch-1.9.1%2Bcu102-cp37-cp37m-linux_x86_64.whl
   pip install torch-1.9.1+cu102-cp37-cp37m-linux_x86_64.whl
   
   wget https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
   pip install torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
   
   wget https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
   pip install torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
   
   pip install torch-geometric==2.0.2
```

## Dataset Preparation

This project supports two datasets: **MGTAB** and ​**TwiBot22**​. Prepare the datasets as follows:

### 1. MGTAB Dataset

* Download the preprocessed `.pt` files from the official [MGTAB repository](https://github.com/GraphDetec/MGTAB/blob/main/Dataset/MGTAB).
* Place all files into: Dataset/MGTAB/

### 2. TwiBot22 Dataset

* Generate dataset files using the code from [TwiBot-22/src/BotRGCN](https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/BotRGCN).
* After generation, ensure the following files are available: train_idx.pt, test_idx.pt,
  val_idx.pt, label.pt, edge_index.pt, edge_type.pt, num_properties_tensor.pt, cat_properties_tensor.pt, tweets_tensor.pt, des_tensor.pt
* Place all files into: Dataset/TwiBot22/

## Running the Code

Use the following commands to train the model on each dataset:

### For MGTAB

```bash
   python train.py --dataset mgtab --epochs 200 --lr 1e-3 --dropout 0.1
```

### For TwiBot22
```bash
   python train.py --dataset twibot22 --epochs 200 --lr 1e-3 --dropout 0.5
```
