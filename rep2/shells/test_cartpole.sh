#!/bin/bash

echo "🚀 Test is started at  : $(date +'%Y:%m:%d-%H:%M:%S')"
python3 src/test_cartpole.py
echo "🚀 Test is finished at  : $(date +'%Y:%m:%d-%H:%M:%S')"

