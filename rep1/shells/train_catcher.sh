#!/bin/bash

echo "🚀 Train is started at  : $(date +'%Y:%m:%d-%H:%M:%S')"
python3 src/train_catcher.py
echo "🚀 Train is finished at  : $(date +'%Y:%m:%d-%H:%M:%S')"

