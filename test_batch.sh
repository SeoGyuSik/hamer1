#!/bin/bash

echo "###"
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID

python demo_batch.py \
    --img_folder example_data --out_folder demo_out \
    --image_batch_size=2 --side_view --save_mesh --full_frame

echo "###"

