#!/usr/bin/env python3
"""
GraphSAGE Training Script for GraphPCB Dataset

Usage:
    python graphsage.py --dataset Graph-F --aggr softmax --epochs 200
    python graphsage.py --dataset Graph-W --aggr mean --epochs 100 --lr 0.001
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import aggr

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from utils.base_models import GraphSAGE
from utils.logger import PCB_Logger
from utils.utils import set_seed, compute_loss, compute_metrics


class GraphDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        """
        Args:
            dataset_dir (str): Directory containing graph data files.
            transform (callable, optional): Transform to apply to each graph.
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.graph_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pt')]
        self.transform = transform

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        graph_file = self.graph_files[idx]
        graph_path = os.path.join(self.dataset_dir, graph_file)
        data = torch.load(graph_path)
        
        if self.transform:
            data = self.transform(data)
            
        return data


def get_data_loaders(dataset_name, dataset_dir, batch_size=1):
    """Load train and test data loaders for the specified dataset."""
    if dataset_name.lower() == 'wacv':
        train_dir = os.path.join(dataset_dir, 'Graph-W', 'graphs', 'train')
        test_dir = os.path.join(dataset_dir, 'Graph-W', 'graphs', 'test')
    elif dataset_name.lower() == 'fpic':
        train_dir = os.path.join(dataset_dir, 'Graph-F', 'graphs', 'train')
        test_dir = os.path.join(dataset_dir, 'Graph-F', 'graphs', 'test')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_dataset = GraphDataset(train_dir)
    test_dataset = GraphDataset(test_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def get_aggregation(aggr_type):
    """Get the aggregation function based on the type."""
    if aggr_type == "mean":
        return "mean"
    elif aggr_type == "max":
        return "max"
    elif aggr_type == "sum":
        return "sum"
    elif aggr_type == "std":
        return "std"
    elif aggr_type == "softmax":
        return aggr.SoftmaxAggregation(learn=True)
    elif aggr_type == "attn":
        return aggr.AttentionalAggregation(gate_nn=torch.nn.Linear(256, 1))
    elif aggr_type == "multi":
        return aggr.MultiAggregation(['mean', 'std', aggr.SoftmaxAggregation(learn=True)])
    else:
        raise ValueError(f"Unknown aggregation type: {aggr_type}")


def create_graphsage_model(config):
    """Create and return GraphSAGE model."""
    aggr_func = get_aggregation(config['aggr_type'])
    
    return GraphSAGE(
        in_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        out_dim=config['output_dim'],
        num_layers=config.get('num_layers', 2),
        aggr=aggr_func,
        dropout=config['dropout'],
        use_batchnorm=config['use_batchnorm'],
        use_bias=config['use_bias']
    )


def train_model(model, config):
    """Train the GraphSAGE model with the given configuration."""
    set_seed(config["seed"])
    train_loader, test_loader = get_data_loaders(config['dataset'], config['dataset_dir'], batch_size=config["batch_size"])

    logger = PCB_Logger(config=config)

    torch.cuda.empty_cache()

    # Move model to device
    model = model.to(config["device"])
    
    # Define optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler"]["step_size"], gamma=config["scheduler"]["gamma"])

    # Training Loop
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        num_graphs = 0

        for batch in train_loader:
            batch = batch.to(config["device"])
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch.x, batch.edge_index, batch.batch)
            
            # Compute loss
            loss = compute_loss(logits, batch.y, num_classes=config["output_dim"])
            
            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_graphs += batch.num_graphs

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
            predictions = {}

            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(config["device"])
                    logits = model(batch.x, batch.edge_index, batch.batch)
                    preds = torch.argmax(logits, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())

            # Compute metrics
            metrics = compute_metrics(all_preds, all_labels)
            logger.update_metrics(metrics, predictions)
    
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
    parser = argparse.ArgumentParser(description='GraphSAGE Training with GraphPCB Dataset')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True, choices=['Graph-W', 'Graph-F', 'wacv', 'fpic'],
                       help='Dataset to use (Graph-W/wacv for WACV dataset, Graph-F/fpic for FPIC dataset)')
    
    # GraphSAGE specific arguments
    parser.add_argument('--aggr', type=str, default='mean', 
                       choices=['mean', 'max', 'sum', 'std', 'softmax', 'attn', 'multi'],
                       help='Aggregation function for GraphSAGE')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--seed', type=int, default=37, help='Random seed')
    
    # Model architecture arguments
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of GraphSAGE layers')
    
    # Regularization arguments
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--no-batchnorm', action='store_true', help='Disable batch normalization')
    parser.add_argument('--use-bias', action='store_true', help='Use bias in layers')
    
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
        "experiment_name": f"GraphSAGE_{args.aggr}_{dataset_name}",
        "dataset_dir": dataset_dir,
        "home_dir": home_dir,
        "dataset": dataset_name,
        "device": args.device,
        "seed": args.seed,
        "batch_size": args.batch_size,

        # model architecture
        "model": "GraphSAGE",
        "input_dim": 1024,
        "hidden_dim": args.hidden_dim,
        "output_dim": 4,
        "num_layers": args.num_layers,
        "aggr_type": args.aggr,

        # regularization
        "dropout": args.dropout,
        "use_batchnorm": not args.no_batchnorm,
        "use_bias": args.use_bias,
        "weight_decay": args.weight_decay,
        "scheduler": {"type": "StepLR", "step_size": 30, "gamma": 0.5},

        # training parameters
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
    }

    print(f"Starting GraphSAGE training with {args.aggr} aggregation on dataset {dataset_name}")
    print(f"Configuration: {config}")

    # Create and train model
    model = create_graphsage_model(config)
    model, metrics, checkpoint_dir = train_model(model, config)

    print(f"Training completed! Results saved to: {checkpoint_dir}")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()