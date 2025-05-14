# Codes for The GraphPCB Dataset


The full dataset is hosted at Kaggle [The GraphPCB Dataset](https://www.kaggle.com/datasets/irislan/graphpcb)

# Requirements

```
pip install -r requirements.txt
```

# 📁 Folder Structure

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

# Citation