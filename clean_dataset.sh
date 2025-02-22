#!/bin/bash

# Clean Dataset Script
# Usage: ./cleandataset.sh

# Install required libraries (only needed once)
pip3 install python-magic Pillow

# Find and log corrupted images
echo "Scanning for corrupted images..."
find dataset/ -name "*.jpg" -type f -print0 | xargs -0 -P8 -I{} python3 -c "
import sys
from PIL import Image
try:
    with Image.open(sys.argv[1]) as img:
        img.verify()
except Exception as e:
    print(sys.argv[1])
" {} > corrupted_files.log

# Remove corrupted files (safety first!)
if [ -s corrupted_files.log ]; then
    echo "Found $(wc -l < corrupted_files.log) corrupted files. Removing..."
    xargs -a corrupted_files.log rm -v
else
    echo "No corrupted files found!"
fi

# Cleanup
rm -f corrupted_files.log
echo "Dataset cleaning complete!"
