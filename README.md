# Codes for The GraphPCB Dataset

## About The GraphPCB Dataset
The GraphPCB Dataset consists of graph-based representations of printed circuit boards (PCBs), created from two sources: [WACV 2019 PCB Dataset](https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection) and [FPIC-PCB dataset](https://datasetninja.com/fpic-component). Each PCB image is converted into a graph where components are nodes, and spatial relationships are edges. This structure is useful for tasks such as graph classification and component detection in electronic hardware.

### 📈 Dataset Statistics (Graph-W)
| Graph-W    | Train  | Test  | Total  | Ratio |
| --------------------------------- | ------ | ----- | ------ | ----- |
| Number of Graphs               | 37     | 10    | 47     | –     |
| IC (Class 0)                   | 247    | 139   | 386    | 2.0%  |
| Discrete Transformer (Class 1) | 43     | 41    | 84     | 0.5%  |
| Diode (Class 2)                | 53     | 24    | 77     | 0.4%  |
| Others (Class 3)               | 12,790 | 4,864 | 17,654 | 97.0% |


### 📈 Dataset Statistics (Graph-F)
| Graph-F                         | Train  | Test  | Total  | Ratio |
| --------------------------------- | ------ | ----- | ------ | ----- |
| Number of Graphs               | 115    | 47    | 162    | –     |
| IC (Class 0)                   | 1,059  | 227   | 1,286  | 5.1%  |
| Discrete Transformer (Class 1) | 768    | 52    | 820    | 3.0%  |
| Diode (Class 2)                | 421    | 40    | 461    | 1.7%  |
| Others (Class 3)               | 20,399 | 3,499 | 23,898 | 90.2% |




## Download Dataset
The `*.pt` files are included in the repo under `./data/GraphPCB/`. The full dataset, including the source images and masks, is hosted at Kaggle: [The GraphPCB Dataset](https://www.kaggle.com/datasets/irislan/graphpcb).

## Requirements
To build the environment, run

```
conda env create -f environment.yml
```

## 📁 Folder Structure

```
GraphPCB/
├── utils/                      # All the models, utility modules and helper functions
├── data/
│   └── GraphPCB/               # GraphPCB dataset
│       ├── Graph-F/graphs/     # Graph-F dataset (*.pt files)
│       └── Graph-W/graphs/     # Graph-W dataset (*.pt files)
├── node_classification.py      # Command-line script for MLP, GCN, GIN, GAT models
├── graphsage.py                # Command-line script for GraphSAGE model
├── acmgnn.py                   # Command-line script for ACMGNN model
├── graphsep.py                 # Command-line script for GraphSep models
├── acmgnn.ipynb                # [DEPRECATED] Jupyter notebook for ACMGNN
├── graphsep.ipynb              # [DEPRECATED] Jupyter notebook for GraphSep
├── node_classification_GraphPCB.ipynb      # [DEPRECATED] Jupyter notebook for basic models
└── README.md                   
```

## 🚀 Command Line Usage

### Basic Models (MLP, GCN, GIN, GAT)
```bash
# Train GCN on Graph-F dataset
python node_classification.py --dataset Graph-F --model GCN

# Train GAT on Graph-W dataset with custom parameters
python node_classification.py --dataset Graph-W --model GAT --epochs 200 --lr 0.001 --hidden-dim 512
```

### GraphSAGE Model
```bash
# Train GraphSAGE with softmax aggregation
python graphsage.py --dataset Graph-F --aggr softmax --epochs 200

# Train GraphSAGE with mean aggregation and custom parameters
python graphsage.py --dataset Graph-W --aggr mean --epochs 100 --lr 0.001 --num-layers 3
```

### ACMGNN Model
```bash
# Train ACMGNN on Graph-F dataset
python acmgnn.py --dataset Graph-F

# Train ACMGNN with custom parameters
python acmgnn.py --dataset Graph-W --epochs 100 --lr 0.0001 --hidden-dim 128
```

### GraphSep Models
```bash
# Train GT-sep model
python graphsep.py --dataset Graph-F --model gt-sep

# Train GAT-sep model with custom parameters
python graphsep.py --dataset Graph-W --model gat-sep --epochs 150 --num-layers 3
```

### Available Options

**Common Arguments:**
- `--dataset`: Choose between `Graph-W`/`wacv` (WACV dataset) or `Graph-F`/`fpic` (FPIC dataset)
- `--epochs`: Number of training epochs (default: 200)
- `--lr`: Learning rate (default varies by model)
- `--hidden-dim`: Hidden dimension size (default: 256)
- `--dropout`: Dropout rate
- `--weight-decay`: Weight decay for regularization
- `--device`: Device to use (`cuda:0`, `cpu`)
- `--data-dir`: Path to data directory (default: `~/GraphPCB_Analysis/data/GraphPCB`)

**Model-Specific Arguments:**
- `node_classification.py`: `--model` {MLP, GCN, GIN, GAT}, `--num-layers`, `--num-heads` (for GAT)
- `graphsage.py`: `--aggr` {mean, max, sum, std, softmax, attn, multi}
- `acmgnn.py`: `--model` {acmgcn, acmsgc, gcn, sgc, mlp}
- `graphsep.py`: `--model` {gt-sep, gat-sep, gt, gat, gcn, sage, resnet}, `--num-layers`, `--num-heads`

## Citation
