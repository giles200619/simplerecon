#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 22:46:23 2023

@author: jj
"""

import numpy as np
import open3d as o3d
import os
import shutil
import copy
import argparse
import time
    
def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument(
        "--seq_name",
        default='galaxy_1',
        help="folder name to the input data, should include /images /intrinsics /poses"
    )
    parser.add_argument(
        "--data_path",
        default='/home/jj/work/simplerecon/data/scans/',
        help="data path"
    )
    parser.add_argument(
        "--path_simplerecon",
        default='/home/jj/work/simplerecon/',
        help="directory to simplerecon code"
    )
    parser.add_argument('--use_simplerecon_pcd', action='store_true', help='use fused pcd instead of mesh')
    
    return parser

def run_floor_est_gravity(args,n_pts_mesh=30000,dy_check=0.01,vis=False):
    seq_name = args.seq_name 
    base_path = f'{args.data_path}/{seq_name}' 
    
    save_to_path = os.path.join(base_path,'viola')
    if os.path.exists(save_to_path):
        #os.chmod(save_to_path, 0o777)
        shutil.rmtree(save_to_path)

    os.makedirs(save_to_path,exist_ok=True)
    #os.chmod(save_to_path, 0o777)

    if not args.use_simplerecon_pcd:
        mesh = o3d.io.read_triangle_mesh(f'{args.path_simplerecon}/output/HERO_MODEL/viola/default/meshes/0.04_3.0_open3d_color/{seq_name}.ply')
        pcd = mesh.sample_points_uniformly(number_of_points=n_pts_mesh)
        shutil.copy(f'{args.path_simplerecon}/output/HERO_MODEL/viola/default/meshes/0.04_3.0_open3d_color/{seq_name}.ply', save_to_path)
    else:
        pcd = o3d.io.read_point_cloud(f'{args.path_simplerecon}/output/HERO_MODEL/viola/default/pcs/3_0.04_0.02_3.0/{seq_name}.ply')
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        shutil.copy(f'{args.path_simplerecon}/output/HERO_MODEL/viola/default/pcs/3_0.04_0.02_3.0/{seq_name}.ply', save_to_path)
    
    #K = np.loadtxt(f'{base_path}/intrinsics/00000.txt')
    
    Ts = []
    with open(f'./data_splits/{seq_name}/test_eight_view_deepvmvs.txt') as f:
        inds = f.readlines()
    for i in range(len(inds)):
        img_name = inds[i].split(' ')[1]
        Ts.append(np.loadtxt(f'{base_path}/poses/{img_name}.txt'))
    Ts = np.asarray(Ts)
    
    w_T_cam0 = np.loadtxt(f'{base_path}/w_T_cam0.txt')
    
    
    pcd_aligned = copy.copy(pcd)
    pcd_aligned.transform(w_T_cam0)
    cl, ind = pcd_aligned.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
    pcd_aligned = pcd_aligned.select_by_index(ind)
    pcd_aligned_pts = np.array(pcd_aligned.points)
    max_c =0
    dy = 0
    t0 = time.time()
    print('finding floor plane along world y axis..')
    for i in range(int((max(pcd_aligned_pts[:,1])-min(pcd_aligned_pts[:,1]))/dy_check)+1):
        y = min(pcd_aligned_pts[:,1]) + dy_check*i
        if sum(abs(pcd_aligned_pts[:,1]-y)<dy_check) > max_c:
            max_c = sum(abs(pcd_aligned_pts[:,1]-y)<dy_check)
            dy = y 
    print('done',time.time()-t0)
    axis_aligned_T_cam0 = copy.copy(w_T_cam0)
    axis_aligned_T_cam0 = np.linalg.inv(axis_aligned_T_cam0)
    axis_aligned_T_cam0[1,-1] += dy
    axis_aligned_T_cam0 = np.linalg.inv(axis_aligned_T_cam0)
    axis_aligned_T_cam0 = np.array([[0,0,1,0],
                                    [1,0,0,0],
                                    [0,1,0,0],
                                    [0,0,0,1]]) @ axis_aligned_T_cam0
    # save
    data = {}
    data['pts'] = np.asarray(pcd.points)
    data['colors'] = np.asarray(pcd.colors)
    data['m2f_seg'] = None
    data['w_T_cam0'] = Ts[0]
    data['w_T_cams'] = Ts
    data['axis_aligned_T_w'] = axis_aligned_T_cam0
    np.save(os.path.join(save_to_path, 'viola_input.npy'),data)
    
    if vis:
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame(5)
        pcd.transform(axis_aligned_T_cam0)
        o3d.visualization.draw_geometries([pcd, cf])

if __name__ == "__main__":
    args = get_parser().parse_args()    
    
    run_floor_est_gravity(args,n_pts_mesh=30000)
    
