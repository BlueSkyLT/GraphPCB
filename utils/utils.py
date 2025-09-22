import os
import yaml
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import torch.nn.functional as F
from collections import defaultdict
import json
import random
import shutil


def get_parameter_groups(model):
    no_weight_decay_names = ['bias', 'normalization', 'label_embeddings']

    parameter_groups = [
        {
            'params': [param for name, param in model.named_parameters()
                       if not any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)]
        },
        {
            'params': [param for name, param in model.named_parameters()
                       if any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)],
            'weight_decay': 0
        },
    ]

    return parameter_groups


def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None, warmup_proportion=None,
                                 last_step=-1):

    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError('Either num_warmup_steps or num_steps and warmup_proportion should be provided.')

    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

    def get_lr_multiplier(step):
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        else:
            return 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=last_step)

    return lr_scheduler

def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensures deterministic behavior for some operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Calculate the F1 when ignoring the last class
# suppose we have a confusion matrix = [[91, 4, 0, 44], [8, 16, 0, 17], [4, 3, 4, 13], [267, 73, 59, 4465]]
# we should calculate the F1 score as if the matrix was [[91, 4, 0], [8, 16, 0], [4, 3, 4]]
def calculate_subset_metrics(conf_matrix, num_classes=3):
    """
    Calculate the precision, recall, and F1 score for a confusion matrix with 3 classes, ignoring the last class.
    Args:
        conf_matrix (2D array-like): The confusion matrix to calculate the F1 score from.
    Returns:
        precision (float): avg precision for the first 3 classes.
        recall (float): avg recall for the first 3 classes.
        f1_score (float): avg F1 score for the first 3 classes.
    """

    conf_matrix = np.array(conf_matrix)  # Ensure the confusion matrix is a numpy array
    conf_matrix = conf_matrix[:num_classes, :num_classes]  
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    precision = np.mean(TP / (TP + FP)) if np.sum(TP + FP) > 0 else 0
    recall = np.mean(TP / (TP + FN)) if np.sum(TP + FN) > 0 else 0

    f1_score = np.mean(2 * (precision * recall) / (precision + recall)) if precision + recall > 0 else 0
    return f1_score

def compute_loss(logits, y, num_classes=4, gamma=2.0, temperature=0.02, loss_weights=[1.0, 0.1]):
    """
    Compute the loss for the model.

    Args:
        data (torch_geometric.data.Data): Input graph data.
        model_out (tuple): Tuple containing embeddings and logits.
        train_mask (torch.Tensor): Mask for training nodes.
        loss (str): Loss type ("Focal", "NLL", "Contrast", "multi").
        num_classes (int): Number of classes.
        gamma (float): Focusing parameter for Focal Loss.
        temperature (float): Temperature for Contrastive Loss.
        loss_weights (list): Weights for Focal and Contrastive Losses [w_focal, w_contrast].

    Returns:
        dict: Dictionary containing the total loss.
    """
    log_probs = F.log_softmax(logits, dim=1)

    # Compute class weights based on dataset distribution
    class_counts = torch.bincount(y, minlength=num_classes).float()
    weights = torch.where(class_counts > 0, 1.0 / class_counts, torch.tensor(0.0, device=y.device))
    weights = weights / weights.sum() * num_classes  # Normalize
    weights = weights.to(y.device)

    # Initialize total loss
    total_loss = F.nll_loss(log_probs, y, weight=weights)

    return total_loss

def compute_metrics(all_preds, all_labels):
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Display formatting
    f1_3_class = calculate_subset_metrics(cm, num_classes=3)

    return {
        "f1_macro": float(f1),
        "weighted_f1": float(weighted_f1),
        "f1_3_class": float(f1_3_class),
        "f1_per_class": f1_per_class.tolist(),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "confusion_matrix": cm.tolist(),
    }

