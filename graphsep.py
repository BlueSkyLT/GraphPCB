#!/usr/bin/env python3
"""
GraphSep Training Script for GraphPCB Dataset

Usage:
    python graphsep.py --dataset Graph-F --model gt-sep
    python graphsep.py --dataset Graph-W --model gat-sep --epochs 100 --lr 0.0001
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import dgl
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from utils.graphsep_model import Model
from utils.logger import PCB_Logger
from utils.utils import set_seed, get_parameter_groups, get_lr_scheduler_with_warmup, compute_metrics


class GraphPCBDataset:
    def __init__(self, data_dir, device='cpu', add_self_loops=False, to_bidirectional=True):
        """
        Initialize the dataset and convert graphs into DGLGraph format.

        Args:
            data_dir (str): Directory containing the `.pt` files with torch_geometric Data objects.
            device (str): Device to load the data onto ('cpu' or 'cuda').
            add_self_loops (bool): Whether to add self-loops to the graphs.
            to_bidirectional (bool): Whether to convert graphs to bidirectional.
        """
        print("Loading dataset...")
        self.device = device
        self.graphs = []
        self.graph_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        # randomly shuffle the graph files
        np.random.shuffle(self.graph_files)
        self.num_node_features = None
        self.num_classes = None
        self.size = len(self.graph_files)

        # Load and convert each graph
        for filename in self.graph_files:
            if filename.endswith(".pt"):
                data = torch.load(os.path.join(data_dir, filename))
                # get the number of node features and the number of classes
                self.num_node_features = data.x.size(1)
                self.num_targets = data.y.max().item() + 1 if data.y is not None else 0

                # Convert edge_index to DGL format
                edge_index = data.edge_index
                num_nodes = data.x.size(0)

                # Create DGL graph
                g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes, device=device)

                if to_bidirectional:
                    g = dgl.to_bidirected(g)
                if add_self_loops:
                    g = dgl.add_self_loop(g)

                # Add node features and labels
                g.ndata['x'] = data.x.to(device)
                g.ndata['y'] = data.y.to(device)

                self.graphs.append(g)

        print(f"Loaded {len(self.graphs)} graphs with {self.num_node_features} node features and {self.num_targets} classes.")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def compute_loss(logits, labels, loss_type="NLL", gamma=2.0):
    """Compute loss function."""
    if loss_type == "NLL":
        return F.cross_entropy(logits, labels)
    elif loss_type == "Focal":
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma) * ce_loss
        return focal_loss.mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_model(model, config):
    """Train the GraphSep model with the given configuration."""
    set_seed(42)
    train_dir = os.path.join(config['dataset_dir'], f"Graph-{config['dataset'].upper()[0]}/graphs", "train")
    test_dir = os.path.join(config['dataset_dir'], f"Graph-{config['dataset'].upper()[0]}/graphs", "test")
    train_dataset = GraphPCBDataset(data_dir=train_dir, device=config["device"], add_self_loops=False)
    test_dataset = GraphPCBDataset(data_dir=test_dir, device=config["device"], add_self_loops=False)

    logger = PCB_Logger(config=config)

    torch.cuda.empty_cache()
    # Move model to device
    model = model.to(config["device"])
    
    # Define optimizer & scheduler
    parameter_groups = get_parameter_groups(model)
    optimizer = torch.optim.AdamW(parameter_groups, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = GradScaler(enabled=config.get('amp', False))
    scheduler = get_lr_scheduler_with_warmup(
        optimizer=optimizer, 
        num_warmup_steps=config.get('num_warmup_steps'),
        num_steps=config['num_epochs'], 
        warmup_proportion=config.get('warmup_proportion', 0.1)
    )

    # Training Loop
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        num_graphs = 0

        for graph in train_dataset:
            optimizer.zero_grad()

            with autocast(enabled=config.get('amp', False)):
                # Forward pass
                logits = model(graph, graph.ndata['x'])
                
                # Compute loss
                loss = compute_loss(logits, graph.ndata['y'], loss_type=config.get('loss', 'NLL'))

            # Backward pass
            if config.get('amp', False):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_graphs += 1

        avg_loss = total_loss / num_graphs
        scheduler.step()

        # Log training progress
        if (epoch + 1) % 10 == 0:
            logger.log(f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {avg_loss:.4f}")

        # Evaluation on test set
        if (epoch + 1) % 50 == 0 or epoch == config["num_epochs"] - 1:
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for graph in test_dataset:
                    logits = model(graph, graph.ndata['x'])
                    preds = torch.argmax(logits, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(graph.ndata['y'].cpu().numpy())

            # Compute metrics
            metrics = compute_metrics(all_preds, all_labels)
            logger.update_metrics(metrics, {})
    
    logger.finish_run()
    # Save final model
    checkpoint_path = os.path.join(logger.checkpoint_dir, f"model_epoch_{epoch + 1}_{avg_loss:.4f}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, checkpoint_path)
    logger.log(f"✅ Model checkpoint saved to {checkpoint_path}")

    return model, metrics, logger.checkpoint_dir


def main():
    parser = argparse.ArgumentParser(description='GraphSep Training with GraphPCB Dataset')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True, choices=['Graph-W', 'Graph-F', 'wacv', 'fpic'],
                       help='Dataset to use (Graph-W/wacv for WACV dataset, Graph-F/fpic for FPIC dataset)')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True, 
                       choices=['gt-sep', 'gat-sep', 'gt', 'gat', 'gcn', 'sage', 'resnet'],
                       help='GraphSep model variant')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model architecture arguments
    parser.add_argument('--num-layers', type=int, default=2, help='Number of model layers')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--hidden-dim-multiplier', type=float, default=1.0, help='Hidden dimension multiplier')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--normalization', type=str, default='LayerNorm', 
                       choices=['None', 'LayerNorm', 'BatchNorm'], help='Normalization type')
    
    # Regularization arguments
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay')
    
    # Training specific arguments
    parser.add_argument('--warmup-proportion', type=float, default=0.1, help='Warmup proportion')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--loss', type=str, default='NLL', choices=['NLL', 'Focal'], help='Loss function')
    
    # Path arguments
    parser.add_argument('--data-dir', type=str, default=None, 
                       help='Path to data directory (default: ~/GraphPCB_Analysis/data/GraphPCB)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (cuda:0, cpu)')

    args = parser.parse_args()

    # Set up paths
    if args.data_dir is None:
        home_dir = os.path.join(os.path.expanduser("~"), "GraphPCB_Analysis")
        dataset_dir = os.path.join(home_dir, 'data', 'GraphPCB')
    else:
        dataset_dir = args.data_dir
        home_dir = os.path.dirname(os.path.dirname(dataset_dir))

    # Normalize dataset name
    if args.dataset.lower() in ['graph-w', 'wacv']:
        dataset_name = 'wacv'
    elif args.dataset.lower() in ['graph-f', 'fpic']:
        dataset_name = 'fpic'
    else:
        dataset_name = args.dataset.lower()

    # Normalize model name
    model_name = args.model.upper()

    # Create configuration
    config = {
        "experiment_name": f"GraphSep_{model_name}_{dataset_name}",
        "dataset_dir": dataset_dir,
        "home_dir": home_dir,
        "dataset": dataset_name,
        "device": args.device,
        "batch_size": 1,

        # model architecture
        "model": model_name,
        "num_layers": args.num_layers,
        "input_dim": 1024,
        "hidden_dim": args.hidden_dim,
        "hidden_dim_multiplier": args.hidden_dim_multiplier,
        "output_dim": 4,
        "num_heads": args.num_heads,
        "normalization": args.normalization,

        # regularization
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,

        # training parameters
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "num_warmup_steps": None,
        "warmup_proportion": args.warmup_proportion,
        "amp": args.amp,
        "loss": args.loss,
    }

    print(f"Starting GraphSep training with model {model_name} on dataset {dataset_name}")
    print(f"Configuration: {config}")

    # Create and train model
    model = Model(
        model_name=model_name,
        num_layers=args.num_layers,
        input_dim=1024,
        hidden_dim=args.hidden_dim,
        output_dim=4,
        hidden_dim_multiplier=args.hidden_dim_multiplier,
        num_heads=args.num_heads,
        normalization=args.normalization,
        dropout=args.dropout
    )
    
    model, metrics, checkpoint_dir = train_model(model, config)

    print(f"Training completed! Results saved to: {checkpoint_dir}")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()