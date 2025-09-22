#!/bin/bash

# GraphPCB Model Evaluation Script
# This script runs all models with optimal settings and displays results in a table format

echo "=========================================="
echo "GraphPCB Model Evaluation Suite"
echo "=========================================="
echo ""

# Set dataset - using WACV (Graph-W) as default based on most notebook examples
DATASET="Graph-W"
RESULTS_FILE="/tmp/model_results.txt"

# Clear previous results
> $RESULTS_FILE

echo "Running models on dataset: $DATASET"
echo "This may take a while..."
echo ""

# Function to extract metrics from log file
extract_metrics() {
    local model_name=$1
    local log_file=$2
    
    if [ -f "$log_file" ]; then
        # Extract F1-Score (macro) and Subset F1-Score (3-class)
        f1_macro=$(grep "F1-Score (macro):" "$log_file" | tail -1 | sed 's/.*F1-Score (macro): //' | awk '{print $1}')
        f1_3class=$(grep "Subset F1-Score (3-class):" "$log_file" | tail -1 | sed 's/.*Subset F1-Score (3-class): //' | awk '{print $1}')
        
        # Extract POD values for labels 0, 1, 2 (only for non-MLP models)
        if [ "$model_name" != "MLP" ]; then
            pod_line=$(grep "POD per label:" "$log_file" | tail -1)
            pod_0=$(echo "$pod_line" | grep -o "0: [0-9.]*" | cut -d' ' -f2)
            pod_1=$(echo "$pod_line" | grep -o "1: [0-9.]*" | cut -d' ' -f2)
            pod_2=$(echo "$pod_line" | grep -o "2: [0-9.]*" | cut -d' ' -f2)
            
            echo "$model_name|$f1_macro|$f1_3class|$pod_0|$pod_1|$pod_2" >> $RESULTS_FILE
        else
            echo "$model_name|$f1_macro|$f1_3class|-|-|-" >> $RESULTS_FILE
        fi
    else
        echo "Warning: Log file not found for $model_name: $log_file"
        echo "$model_name|ERROR|ERROR|ERROR|ERROR|ERROR" >> $RESULTS_FILE
    fi
}

# Function to get the most recent log file for a model
get_log_file() {
    local model_pattern=$1
    local log_dir="$HOME/GraphPCB_Analysis/Graph-W-trained"
    
    # Find the most recent directory matching the pattern
    latest_dir=$(find "$log_dir" -maxdepth 1 -type d -name "*$model_pattern*" 2>/dev/null | sort | tail -1)
    
    if [ -n "$latest_dir" ]; then
        echo "$latest_dir/train_log.txt"
    else
        echo ""
    fi
}

echo "1/8 Running MLP..."
python node_classification.py --dataset $DATASET --model MLP --hidden-dim 256 --num-layers 2 --dropout 0.3 --lr 0.0001 --epochs 200 --no-batchnorm --weight-decay 1e-3 >/dev/null 2>&1
MLP_LOG=$(get_log_file "MLP")
extract_metrics "MLP" "$MLP_LOG"

echo "2/8 Running GCN..."
python node_classification.py --dataset $DATASET --model GCN --hidden-dim 256 --num-layers 2 --dropout 0.3 --lr 0.001 --epochs 200 --weight-decay 1e-3 >/dev/null 2>&1
GCN_LOG=$(get_log_file "GCN")
extract_metrics "GCN" "$GCN_LOG"

echo "3/8 Running GAT..."
python node_classification.py --dataset $DATASET --model GAT --hidden-dim 1024 --num-layers 2 --num-heads 4 --dropout 0.3 --lr 0.0001 --epochs 200 --weight-decay 1e-3 >/dev/null 2>&1
GAT_LOG=$(get_log_file "GAT")
extract_metrics "GAT" "$GAT_LOG"

echo "4/8 Running GIN..."
python node_classification.py --dataset $DATASET --model GIN --hidden-dim 256 --num-layers 2 --dropout 0.3 --lr 0.0001 --epochs 200 --weight-decay 1e-3 >/dev/null 2>&1
GIN_LOG=$(get_log_file "GIN")
extract_metrics "GIN" "$GIN_LOG"

echo "5/8 Running GraphSAGE (with softmax)..."
python graphsage.py --dataset $DATASET --aggr softmax --num-layers 2 --hidden-dim 256 --dropout 0.3 --lr 0.0001 --epochs 200 --weight-decay 1e-3 >/dev/null 2>&1
SAGE_LOG=$(get_log_file "GraphSAGE")
extract_metrics "GraphSAGE" "$SAGE_LOG"

echo "6/8 Running ACMGNN (with acmgcn)..."
python acmgnn.py --dataset $DATASET --model acmgcn --hidden-dim 256 --dropout 0.5 --lr 1e-4 --epochs 200 --weight-decay 1e-2 >/dev/null 2>&1
ACMGNN_LOG=$(get_log_file "ACMGNN")
extract_metrics "ACMGNN" "$ACMGNN_LOG"

echo "7/8 Running GT-sep..."
python graphsep.py --dataset $DATASET --model gt-sep --num-layers 2 --hidden-dim 256 --num-heads 8 --dropout 0.5 --lr 1e-4 --epochs 200 --weight-decay 1e-2 >/dev/null 2>&1
GTSEP_LOG=$(get_log_file "GT-sep")
extract_metrics "GT-sep" "$GTSEP_LOG"

echo "8/8 Running GAT-sep..."
python graphsep.py --dataset $DATASET --model gat-sep --num-layers 2 --hidden-dim 256 --num-heads 8 --dropout 0.5 --lr 1e-4 --epochs 200 --weight-decay 1e-2 >/dev/null 2>&1
GATSEP_LOG=$(get_log_file "GAT-sep")
extract_metrics "GAT-sep" "$GATSEP_LOG"

echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
echo ""

# Print results table
printf "%-12s | %-12s | %-12s | %-8s | %-8s | %-8s\n" "Model" "F1-Score" "Subset F1" "POD(IC)" "POD(DT)" "POD(Diode)"
printf "%-12s-+-%-12s-+-%-12s-+-%-8s-+-%-8s-+-%-8s\n" "------------" "------------" "------------" "--------" "--------" "--------"

while IFS='|' read -r model f1_macro f1_3class pod_0 pod_1 pod_2; do
    printf "%-12s | %-12s | %-12s | %-8s | %-8s | %-8s\n" "$model" "$f1_macro" "$f1_3class" "$pod_0" "$pod_1" "$pod_2"
done < $RESULTS_FILE

echo ""
echo "Notes:"
echo "- F1-Score: Macro-averaged F1 score across all classes"
echo "- Subset F1: F1 score for the first 3 classes (IC, DT, Diode)"
echo "- POD: Percentage of Overlapping Detections compared to MLP baseline"
echo "- IC: Integrated Circuits (label 0)"
echo "- DT: Discrete Components (label 1)" 
echo "- Diode: Diode components (label 2)"
echo ""
echo "Detailed logs are available in ~/GraphPCB_Analysis/Graph-W-trained/<model_name>/"

# Clean up
rm -f $RESULTS_FILE