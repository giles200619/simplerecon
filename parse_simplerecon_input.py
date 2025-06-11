#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:20:45 2023

@author: jj
"""
import os
import numpy as np
from io import StringIO
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from tqdm import trange
import glob
from PIL import Image
from natsort import natsorted
import argparse
import shutil


galaxy_intrinsics = np.array([[479.28787, 0,         322.37076],
                              [0,         476.65683, 238.06183],
                              [0,         0,         1]])


def extract_frames(video_path, out_folder, size=(640,480)):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if ret is not True:
            break
        frame = cv2.resize(frame, size)
        cv2.imwrite(os.path.join(out_folder, str(i).zfill(5) + '.png'), frame)
        
def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument(
        "--scene_name",
        default='arcore-dataset-2023-10-19-14-00-25',
        help="folder name to the APP output"
    )
    parser.add_argument(
        "--data_path",
        default='/labdata/selim/video2floorplan/galaxy_scans/office',
        help="directory to where all recordings are saved"
    )       
    parser.add_argument(
        "--path_simplerecon",
        default='/home/junjee.chao/work/simplerecon/',
        help="directory to simplerecon code"
    )     
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()    
    path_simplerecon = args.path_simplerecon
    
    path = f'{args.data_path}/{args.scene_name}'
    save_to_name = args.scene_name
    
    
    #video_name = glob.glob(f'{path}/*.mp4')[0]
    file_name = glob.glob(f'{path}/*.txt')[0]
    ori_img_path = os.path.join(path,'images')
    
    #save_to = f'{path_simplerecon}/data/scans/{save_to_name}'
    #os.chmod(path, 0o777)
    #os.chmod(file_name, 0o777)
    save_to = path
    save_img_p = os.path.join(save_to,'images')
    save_intr_p = os.path.join(save_to,'intrinsics')
    save_pose_p = os.path.join(save_to,'poses')    
    
    os.makedirs(save_to,exist_ok=True)
    os.makedirs(save_img_p,exist_ok=True)
    os.makedirs(save_intr_p,exist_ok=True)
    os.makedirs(save_pose_p,exist_ok=True)
    # generate data_split files
    os.makedirs(f'{path_simplerecon}/data_splits/{save_to_name}',exist_ok=True)
    with open(f'{path_simplerecon}/data_splits/{save_to_name}/scans.txt', "w") as f:
        f.write(f"{save_to_name}")
    # generate config file
    with open(f'{path_simplerecon}/configs/data/neucon_arkit_default.yaml', "r") as f:
        config_template = f.readlines()
    #config_template[1] = f'dataset_path: {path_simplerecon}/data\n'
    config_template[1] = f'dataset_path: {args.data_path}\n'
    config_template[2] = f'tuple_info_file_location: data_splits/{save_to_name}/\n'
    config_template[3] = f'dataset_scan_split_file: data_splits/{save_to_name}/scans.txt\n'
    config_template[4] = 'dataset: viola\n'
    with open(f'{path_simplerecon}/configs/data/{save_to_name}_default.yaml',"w") as f:
        for line in config_template:
            f.write(line)
        
    config_template[5] = 'mv_tuple_file_suffix: _eight_view_deepvmvs_dense.txt\n'
    config_template[7] = 'frame_tuple_type: dense\n'
    with open(f'{path_simplerecon}/configs/data/{save_to_name}_dense.yaml',"w") as f:
        for line in config_template:
            f.write(line)
    #extract_frames(video_name, save_img_p)
    
    
    s = open(file_name).read().replace(':', ',')
    data = np.loadtxt(StringIO(s), delimiter=',', dtype=str)
    
    positions = []
    rotmats = []
    #translation_indices = [2, 4, 6]
    #quaternion_indices = [9, 11, 13, 15]
    translation_indices = [2+2, 4+2, 6+2]
    quaternion_indices = [9+2, 11+2, 13+2, 15+2]
    
    for data_el in data:
        raw_pose_list = np.array([s.strip("[]") for s in data_el.tolist()])
        positions.append(raw_pose_list[translation_indices].astype(float))
        rotmats.append(R.from_quat(raw_pose_list[quaternion_indices].astype(float)).as_matrix())
    
    positions = np.stack(positions)
    rotmats = np.stack(rotmats)
    
    tforms = np.eye(4)[None].repeat(positions.shape[0], 0)
    tforms[:, :3, -1]  = positions
    tforms[:, :3, :3]  = rotmats # w_T_cam
    
    tforms_rel = np.linalg.inv(tforms[0]) @ tforms 
    w_T_cam0 = tforms[0]
    np.savetxt(os.path.join(save_to,'w_T_cam0.txt'), w_T_cam0, delimiter=' ')
    Ts = []
    
    ori_img_list = natsorted(os.listdir(ori_img_path))
    assert len(ori_img_list) == tforms_rel.shape[0], "Number of images != number of poses"
    for i in trange(tforms_rel.shape[0]):
        pose = tforms_rel[i]
        pose = pose.dot(np.array([
                    [1, 0, 0,0],
                    [0, -1, 0,0],
                    [0, 0, -1,0],
                    [0,0,0,1]
                ]))
        Ts.append(pose)
        #name = f'{i}'.zfill(5)
        name = ori_img_list[i].split('.')[0]
        np.savetxt(os.path.join(save_pose_p,f'{name}.txt'), pose, delimiter=' ')
        np.savetxt(os.path.join(save_intr_p,f'{name}.txt'), galaxy_intrinsics, delimiter=' ')
        
        #Image.open(os.path.join(ori_img_path, ori_img_list[0])).save(os.path.join(save_img_p,f'{name}.png'))
        #shutil.copyfile(os.path.join(ori_img_path, ori_img_list[i]), os.path.join(save_img_p,f'{name}.png'))
        
        
    # run SimpleRecon
    '''
    python ./data_scripts/generate_test_tuples.py  --num_workers 16 --data_config configs/data/galaxy_0_default.yaml
    
    CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
            --output_base_path output_galaxy \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config configs/data/galaxy_1_default.yaml \
            --num_workers 8 \
            --batch_size 2 \
            --fast_cost_volume \
            --run_fusion \
            --depth_fuser open3d \
            --fuse_color --dump_depth_visualization --cache_depths 
    '''
