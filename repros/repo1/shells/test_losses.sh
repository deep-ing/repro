#!/bin/bash

echo "🚀 Train is started at  : $(date +'%Y:%m:%d-%H:%M:%S')"
python3 test/loss_shapes.py
echo "🚀 Train is finished at  : $(date +'%Y:%m:%d-%H:%M:%S')"

