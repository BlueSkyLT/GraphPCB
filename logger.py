import os
import shutil
import json
from collections import defaultdict
import random
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


class PCB_Logger:
    def __init__(self, home_dir, config):
        self.home_dir = home_dir
        self.config = config
        self.checkpoint_dir = self.get_checkpoint_dir(home_dir, config['dataset'], config['experiment_name'])
        self.predictions = None
        # self.verbose = config['args']['verbose']
        self.log_path = os.path.join(self.checkpoint_dir, "train_log.txt")
        self.predictions_final_path = os.path.join(self.checkpoint_dir, "predictions_final.json")

        self.log("Experiment Configuration:")
        for key, value in self.config.items():
            self.log(f"{key}: {value}")
        print(f"Using device: {self.config['device']}")
        print(f"Loading dataset: {self.config['dataset']}")
        print(f'Results will be saved to {self.checkpoint_dir}.')

    def log(self, message):
        with open(self.log_path, "a") as f:
            f.write(message + "\n")
        print(message)

    def update_metrics(self, metrics, predictions):
        self.predictions = predictions
        metrics_lines = [
            f"  F1-Score (macro): {metrics['f1_macro']:.10f}",
            f"  Weighted F1: {metrics['weighted_f1']:.10f}",
            f"  Subset F1-Score (3-class): {metrics['f1_3_class']:.10f}",
            f"  F1 per class: {metrics['f1_per_class']}",
            f"  Precision per class: {metrics['precision_per_class']}",
            f"  Recall per class: {metrics['recall_per_class']}",
            f"  Confusion Matrix:\n{metrics['confusion_matrix']}"
        ]
        self.log("\n".join(metrics_lines))

    def save_predictions(self):
        with open(self.predictions_final_path, "w") as f:
            json.dump(self.predictions, f)
        self.log(f"✅ Predictions saved to {self.predictions_final_path}")

    def calculate_pod(self):
        """
        Calculate the Percentage of Overlapping Detections (POD) between MLP and GNN predictions.
        """
        # Load predictions from JSON files
        mlp_file = f"{self.home_dir}/GraphPCB_Analysis/Graph-{self.config['dataset'][0].upper()}-trained/MLP/predictions_final.json"
        with open(mlp_file, 'r') as f:
            mlp_predictions = json.load(f)

        gnn_predictions = self.predictions

        # Ensure both files have the same graphs
        mlp_graphs = {graph['graph_id']: graph.get('predictions') or graph.get('labels') or graph.get('gt') for graph in mlp_predictions}
        gnn_graphs = {graph['graph_id']: graph.get('predictions') or graph.get('labels') or graph.get('gt') for graph in gnn_predictions}
        assert mlp_graphs.keys() == gnn_graphs.keys(), "Mismatch in graph IDs between MLP and GNN predictions."

        # Initialize metrics
        total_overlap = 0
        total_nodes = 0
        pod_per_graph = []
        label_overlap = defaultdict(int)
        label_total = defaultdict(int)

        # Calculate POD for each graph
        for graph_id in mlp_graphs:
            mlp_preds = mlp_graphs[graph_id]
            gnn_preds = gnn_graphs[graph_id]
            assert len(mlp_preds) == len(gnn_preds), f"Mismatch in number of nodes for graph {graph_id}."

            # Calculate overlap for this graph
            graph_overlap = sum(1 for m, g in zip(mlp_preds, gnn_preds) if m == g)
            graph_total = len(mlp_preds)
            pod_per_graph.append(graph_overlap / graph_total)

            # Update overall metrics
            total_overlap += graph_overlap
            total_nodes += graph_total

            # Update label-wise metrics
            for m, g in zip(mlp_preds, gnn_preds):
                label_total[m] += 1
                if m == g:
                    label_overlap[m] += 1

        # Calculate overall POD
        overall_pod = total_overlap / total_nodes

        # Calculate POD per label
        pod_per_label = {label: (label_overlap[label] / label_total[label] if label_total[label] > 0 else 0)
                        for label in sorted(label_total.keys())}

        # Return metrics
        return {
            "overall_pod": overall_pod,
            "average_pod_per_graph": sum(pod_per_graph) / len(pod_per_graph),
            "pod_per_label": pod_per_label
        }

    def finish_run(self):
        self.save_predictions()
        if self.config["model"].lower() != "mlp":
            pod_metrics = self.calculate_pod()
            self.log("Overall POD: {:.4f}".format(pod_metrics['overall_pod']))
            self.log("Average POD per graph: {:.4f}".format(sum(pod_metrics['pod_per_label'].values()) / len(pod_metrics['pod_per_label'])))
            self.log(f"POD per label: {', '.join(f'{label}: {pod:.4f}' for label, pod in pod_metrics['pod_per_label'].items())}")
        self.log(f"✅ Final training log saved to {self.log_path}")

    @staticmethod
    def get_checkpoint_dir(home_dir, dataset_name, experiment_name):
        checkpoint_dir = os.path.join(home_dir, f'GraphPCB_Analysis/Graph-{dataset_name[0].upper()}-trained/{experiment_name}')
        
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {checkpoint_dir}")

        return checkpoint_dir