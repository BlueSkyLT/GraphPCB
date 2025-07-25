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
├── acmgnn/                     # ACMGNN model
├── data/
│   └── GraphPCB/               # GraphPCB dataset
│       ├── Graph-F/graphs/     # Graph-F dataset (*.pt files)
│       └── Graph-W/graphs/     # Graph-W dataset (*.pt files)
├── graphsep/                   # GT-sep and GAT-sep models
├── acmgnn.ipynb                # script to run ACMGNN on GraphPCB dataset
├── base_models.py              # basic MLP, GCN, GIN, GAT, GraphSAGE
├── graphsep.ipynb              # script to run GT-sep and GAT-sep on GraphPCB dataset
├── logger.py                   # helper functions
├── node_classification_GraphPCB.ipynb      # script to run basic MLP, GCN, GIN, GAT, GraphSAGE on GraphPCB dataset
├── README.md
└── utils.py                    # helper functions
```

## Citation
