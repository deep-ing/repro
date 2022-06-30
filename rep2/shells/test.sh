#!/bin/bash

echo "🚀 Test is started at  : $(date +'%Y:%m:%d-%H:%M:%S')"
python3 src/test.py --checkpoint_path $1
echo "🚀 Test is finished at  : $(date +'%Y:%m:%d-%H:%M:%S')"

