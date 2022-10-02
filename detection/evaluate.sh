#! /bin/zsh

set -e
dir=$(dirname $(readlink -fn --  $0))
cd $dir/..

module_name=$(basename $dir)
experiment_dir=/content/zero_virus/experiments/cam4
dataset_dir=/content/drive/MyDrive/dataset_light

mkdir -p $experiment_dir
# python -m $module_name.utils.run $dataset_dir $experiment_dir/output.txt

# To produce visualization videos
python -m $module_name.utils.visualize $dataset_dir $dataset_dir $experiment_dir
