#!/bin/bash

echo "🚀 Train is started at  : $(date +'%Y:%m:%d-%H:%M:%S')"
python3 src/train_cartpole_baseline.py
echo "🚀 Train is finished at  : $(date +'%Y:%m:%d-%H:%M:%S')"

