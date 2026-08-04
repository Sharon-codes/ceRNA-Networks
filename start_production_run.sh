#!/bin/bash
echo "Starting 100-Epoch Production Training in background..."
nohup python train.py --epochs 100 --batch_size 64 > production_training.log 2>&1 &

echo "Training detached and running in the background."
echo "You can safely close the terminal."
echo "Monitor progress via: tail -f production_training.log"
