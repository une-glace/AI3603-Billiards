#!/bin/bash

echo "=========================================="
echo "Starting Full Pipeline Execution"
echo "=========================================="

echo "[1/3] Collecting Dataset..."
python train/collect_dataset.py --games 200 --samples 50 --out train/dataset_final.npz
if [ $? -ne 0 ]; then
    echo "Error: Data collection failed!"
    exit 1
fi
echo "Data collection completed successfully."

echo "[2/3] Training CQR Model..."
python train/train_cqr.py --data train/dataset_final.npz --out eval/checkpoints/cqr.joblib
if [ $? -ne 0 ]; then
    echo "Error: Model training failed!"
    exit 1
fi
echo "Model training completed successfully."

echo "[3/3] Running Evaluation..."
python evaluate.py
if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed!"
    exit 1
fi

echo "=========================================="
echo "All tasks completed successfully!"
echo "=========================================="
