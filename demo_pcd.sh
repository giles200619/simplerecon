#!/bin/bash

vis=False

while getopts n:v: flag
do
    case "${flag}" in
        n) scene_name=${OPTARG};;
        v) vis=${OPTARG};;
    esac
done

# datapath=/labdata/selim/video2floorplan/galaxy_scans/office
datapath=/home/selim/data/galaxy_note20

sudo chmod -R 777 $datapath
sudo chmod -R 777 /home/junjee.chao/work/simplerecon

configpath=configs/data/aaa_default.yaml
echo "Scene name:" $scene_name;
echo "config file name:" "${configpath/aaa/"$scene_name"}";
eval "$(conda shell.bash hook)"
conda activate /home/junjee.chao/anaconda3/envs/simplerecon/
#:'
# parse input to simplerecon format
python parse_simplerecon_input.py --scene_name $scene_name --data_path $datapath

# run Simplerecon
python ./data_scripts/generate_test_tuples.py  --num_workers 16 --data_config "${configpath/aaa/"$scene_name"}"

CUDA_VISIBLE_DEVICES=3 python pc_fusion.py --name HERO_MODEL \
            --output_base_path output \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config "${configpath/aaa/"$scene_name"}" \
            --num_workers 8 \
            --batch_size 4 
conda deactivate
#'

# result will be saved: ./output/HERO_MODEL/viola/default/meshes/0.04_3.0_open3d_color
# viola
conda activate /home/selim/anaconda3/envs/torch2/
# estimate floor
python process_viola.py --seq_name $scene_name --data_path $datapath --path_simplerecon /home/junjee.chao/work/simplerecon/ --use_simplerecon_pcd


# viola matching
cd /home/junjee.chao/work/video2floorplan/src/
python run_simplerecon.py --scene_name $scene_name --data_path $datapath --path_simplerecon /home/junjee.chao/work/simplerecon/

