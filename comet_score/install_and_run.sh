#!/bin/bash

input_path=$1
output_path=$2
batch_size=$3

pip install "unbabel-comet>=2.1.0"
python3 get_comet_scores.py --input_path=$input_path --output_path=$output_path --batch_size=$batch_size
