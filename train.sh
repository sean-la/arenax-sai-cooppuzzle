#!/bin/bash

set -e

echo "Collecting expert demonstratrations from shortest path algorithm."

python arenax.py demonstrate \
    --num_demonstrations 2000 \
    --maze_path res/maze.csv \
    --demos_path res/demos.npy

echo "Training behavioural clone model"

python arenax.py train \
    --n_epochs 5000 \
    --batch_size -1 \
    --total_timesteps 1000000 \
    --demos_path res/demos.npy \
    --pretrain_save_location res/pretrain.model