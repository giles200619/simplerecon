#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:13:22 2023

@author: jj
"""

import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from tqdm import trange


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline();
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape = (4, 4))
            for i in range(4):
                matstr = f.readline();
                mat[i, :] = np.fromstring(matstr, dtype = float, sep=' \t')
            #traj.append(CameraPose(metadata, mat))
            data = {}
            data['metadata'] = ' '.join(map(str, metadata))
            data['pose'] = mat # w_T_cam
            traj.append(data)
            metastr = f.readline()
    return traj

if __name__ == "__main__":
    source_path = '/home/jj/work/data/our_itw/office/long_sequences/office_7'
    save_to = f'/home/jj/work/simplerecon/data/scans/{os.path.basename(source_path)}' 
    img_resize_to = (640,480)
        
    save_img_p = os.path.join(save_to,'images')
    save_intr_p = os.path.join(save_to,'intrinsics')
    save_pose_p = os.path.join(save_to,'poses')    
    
    os.makedirs(save_to,exist_ok=True)
    os.makedirs(save_img_p,exist_ok=True)
    os.makedirs(save_intr_p,exist_ok=True)
    os.makedirs(save_pose_p,exist_ok=True)
    
    # read intrinsics
    intrinsics = np.eye(3)
    with open('/home/jj/work/data/our_itw/office/calib.txt') as f:
        inds = f.readlines()[0]
        inds = inds.replace('\n','')
    intrinsics[0,0] = inds.split(' ')[0]
    intrinsics[1,1] = inds.split(' ')[1]
    intrinsics[0,2] = inds.split(' ')[2]
    intrinsics[1,2] = inds.split(' ')[3]
        
    # images 
    rgb_ls = os.listdir(os.path.join(source_path,'image'))
    ori_w, ori_h = Image.open(os.path.join(source_path,'image',rgb_ls[0])).size
    
    # resize 
    intrinsics[0, :] /= (ori_w / img_resize_to[0])
    intrinsics[1, :] /= (ori_h / img_resize_to[1])
    
    # poses
    all_poses = read_trajectory(os.path.join(source_path,'scene', 'trajectory.log'))
    for i in trange(len(rgb_ls)):
        pose = all_poses[int(rgb_ls[i].split('.')[0])]['pose']
        img = Image.open(os.path.join(source_path,'image',rgb_ls[i])).resize(img_resize_to)
        name = rgb_ls[i].split('.')[0][1:]
        
        img.save(os.path.join(save_img_p,f'{name}.png'))
        np.savetxt(os.path.join(save_pose_p,f'{name}.txt'), pose, delimiter=' ')
        np.savetxt(os.path.join(save_intr_p,f'{name}.txt'), intrinsics, delimiter=' ')