#!/bin/bash
# AIT102 Training Queue & Auto-Shutdown Script

# Source Conda
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate ait

echo "==========================================="
echo "  Starting Training Queue at $(date)"
echo "==========================================="

# # 1. Picasso
# echo ">>> Starting Picasso Training..."

# python train_manager.py --style picasso
# if [ $? -ne 0 ]; then
#     echo "Picasso failed! Continuing..."
#     # Optionally break or continue based on preference. 
#     # Continuing is safer for overnight runs so others might finish.
# fi

# # 2. Van Gogh
# echo ">>> Starting Van Gogh Training..."
# python train_manager.py --style vangogh
# if [ $? -ne 0 ]; then
#     echo "Van Gogh failed! Continuing..."
# fi

# # 3. Cyberpunk
# echo ">>> Starting Cyberpunk Training..."
# python train_manager.py --style cyberpunk
# if [ $? -ne 0 ]; then
#     echo "Cyberpunk failed! Continuing..."
# fi

# 4. Monet
echo ">>> Starting Monet Training..."
python train_manager.py --style monet
if [ $? -ne 0 ]; then
    echo "Monet failed! Continuing..."
fi

# 5. Seurat
echo ">>> Starting Seurat Training..."
python train_manager.py --style seurat
if [ $? -ne 0 ]; then
    echo "Seurat failed! Continuing..."
fi

# 6. Wu Guanzhong
echo ">>> Starting Wu Guanzhong Training..."
python train_manager.py --style wuguanzhong
if [ $? -ne 0 ]; then
    echo "Wu Guanzhong failed! Continuing..."
fi

# End
echo "==========================================="
echo "  All Training Tasks Completed at $(date)"
echo "==========================================="

# Sync Output to disk just in case
sync

# Auto Shutdown
echo "SHUTTING DOWN SYSTEM NOW..."
sudo shutdown -h 100
