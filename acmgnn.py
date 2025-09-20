#!/usr/bin/env python3
"""
ACMGNN Training Script for GraphPCB Dataset

Usage:
    python acmgnn.py --dataset Graph-F
    python acmgnn.py --dataset Graph-W --epochs 100 --lr 0.0001
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from glob import glob
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from utils.acmgnn_models import GCN
from utils.logger import PCB_Logger
from utils.utils import set_seed, compute_metrics


def get_data_loader(config):
    """Load train and test graph file paths."""
    train_dir = os.path.join(config['dataset_dir'], f"Graph-{config['dataset'][0].upper()}/graphs", "train")
    test_dir = os.path.join(config['dataset_dir'], f"Graph-{config['dataset'][0].upper()}/graphs", "test")
    train_graphs = sorted(glob(f"{train_dir}/*.pt"))
    test_graphs = sorted(glob(f"{test_dir}/*.pt"))
    return train_graphs, test_graphs


def load_our_data(graph_path, device):
    """
    Load user's .pt graph file in PyG Data format and return
    sparse normalized adjacency matrices (low/high frequency),
    features, and labels.
    """
    data = torch.load(graph_path, map_location=device)

    x = data.x.to(device)  # Node features
    y = data.y.to(device)  # Node labels
    edge_index = data.edge_index.to(device)

    num_nodes = x.size(0)

    # Convert edge_index to dense adjacency
    adj_dense = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    adj_dense[adj_dense != 0] = 1
    adj_dense.fill_diagonal_(0)

    # Compute normalized low- and high-pass filters
    adj_low = adj_dense + torch.eye(num_nodes, device=device)
    deg = torch.sum(adj_low, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt.unsqueeze(1) * adj_low * deg_inv_sqrt.unsqueeze(0)
    adj_low = norm.to_sparse()
    
    # High-pass filter
    adj_high = (torch.eye(num_nodes, device=device) - norm).to_sparse()

    return adj_low, adj_high, x, y


def train_model(config):
    """Train the ACMGNN model with the given configuration."""
    set_seed(42)
    train_graphs, test_graphs = get_data_loader(config)

    logger = PCB_Logger(config=config)

    torch.cuda.empty_cache()

    # Load first graph to initialize model
    adj_low, adj_high, features, labels = load_our_data(train_graphs[0], config["device"])
    model = GCN(
        nfeat=features.shape[1],
        nhid=config["hidden_dim"],
        nclass=labels.max().item() + 1,
        dropout=config["dropout"],
        model_type=config["model"]
    ).to(config["device"])

    # Define optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler"]["step_size"], gamma=config["scheduler"]["gamma"])

    # Training Loop
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        num_graphs = 0

        for graph_path in train_graphs:
            adj_low, adj_high, features, labels = load_our_data(graph_path, config["device"])
            
            optimizer.zero_grad()

            # Forward pass
            logits = model(features, adj_low, adj_high)
            
            # Compute loss (using cross-entropy)
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
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
                for graph_path in test_graphs:
                    adj_low, adj_high, features, labels = load_our_data(graph_path, config["device"])
                    logits = model(features, adj_low, adj_high)
                    preds = torch.argmax(logits, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

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
    parser = argparse.ArgumentParser(description='ACMGNN Training with GraphPCB Dataset')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True, choices=['Graph-W', 'Graph-F', 'wacv', 'fpic'],
                       help='Dataset to use (Graph-W/wacv for WACV dataset, Graph-F/fpic for FPIC dataset)')
    
    # ACMGNN specific arguments
    parser.add_argument('--model', type=str, default='acmgcn', 
                       choices=['acmgcn', 'acmsgc', 'gcn', 'sgc', 'mlp'],
                       help='ACMGNN model variant')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model architecture arguments
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension size')
    
    # Regularization arguments
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay')
    
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

    # Create configuration
    config = {
        "experiment_name": f"ACMGNN_{args.model}_{dataset_name}",
        "dataset_dir": dataset_dir,
        "home_dir": home_dir,
        "dataset": dataset_name,
        "device": args.device,

        # model architecture
        "model": args.model,
        "input_dim": 1024,
        "hidden_dim": args.hidden_dim,
        "output_dim": 4,

        # regularization
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "scheduler": {"type": "StepLR", "step_size": 20, "gamma": 0.5},

        # training parameters
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
    }

    print(f"Starting ACMGNN training with model {args.model} on dataset {dataset_name}")
    print(f"Configuration: {config}")

    # Train model
    model, metrics, checkpoint_dir = train_model(config)

    print(f"Training completed! Results saved to: {checkpoint_dir}")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()