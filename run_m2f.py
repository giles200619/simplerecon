#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 18:14:05 2023

@author: jj
"""

import numpy as np
import open3d as o3d
import os
import subprocess
import glob
import gzip
import torch
import shutil
from natsort import os_sorted
from PIL import Image
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from PIL import ImageColor
import copy
import argparse
from tqdm import tqdm, trange

reduced_ = [
        "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
            ]
reduced_colormap = np.asarray([np.asarray(ImageColor.getcolor(x, "RGB"))/255 for x in reduced_])


def filter_small_components(src_mesh):
    triangle_clusters, cluster_n_triangles, cluster_area = src_mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    tgt_mesh = copy.deepcopy(src_mesh)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 5000
    tgt_mesh.remove_triangles_by_mask(triangles_to_remove)

    return tgt_mesh


def get_thing_semantics(sc_classes='extended'):
    thing_semantics = [False]
    for cllist in [x.strip().split(',') for x in Path(f"/home/junjee.chao/work/panoptic-lifting/resources/scannet_{sc_classes}_things.csv").read_text().strip().splitlines()]:
        thing_semantics.append(bool(int(cllist[1])))
    return thing_semantics

def convert_from_mask_to_semantics_and_instances_no_remap(original_mask, segments, _coco_to_scannet, is_thing, instance_ctr, instance_to_semantic):
    id_to_class = torch.zeros(1024).int()
    instance_mask = torch.zeros_like(original_mask)
    invalid_mask = original_mask == 0
    for s in segments:
        id_to_class[s['id']] = s['category_id']
        if is_thing[s['category_id']]:
            instance_mask[original_mask == s['id']] = instance_ctr
            instance_to_semantic[instance_ctr] = s['category_id']
            instance_ctr += 1
    return id_to_class[original_mask.flatten().numpy().tolist()].reshape(original_mask.shape), instance_mask, invalid_mask, instance_ctr, instance_to_semantic

def T_pcd(T,pcd):
    return (T @ np.concatenate((pcd,np.ones((pcd.shape[0],1))),axis=1).T).T[:,:3]

def fuse_m2f(args, seq_name, Ts, img_list, number_of_points=100000, chunck_size=10000, vis=False):    
    
    mesh = o3d.io.read_triangle_mesh(f'{args.data_path}/{seq_name}/viola/{seq_name}.ply')
    filtered_recon = filter_small_components(mesh)
    mesh = filtered_recon
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    pts = np.asarray(pcd.points)
    seg_labels = np.zeros((0,))
    colors = np.zeros((0,3))
    
    rgb_p = f'{args.data_path}/{seq_name}/m2f/rgb/'
    prob_p = f'{args.data_path}/{seq_name}/m2f/mask2former/m2f_probabilities/'
        
    K = np.loadtxt(glob.glob(f'{args.data_path}/{seq_name}/intrinsics/*.txt')[0])
    img = np.asarray(Image.open(glob.glob(os.path.join(rgb_p,'*'))[0]))
    H, W = img.shape[:2]
    
    chunks = int(number_of_points/chunck_size) if number_of_points%chunck_size==0 else int(number_of_points/chunck_size)+1
    for chunk in trange(chunks):
        
        n_frames = Ts.shape[0]
        full_pts = pts[chunck_size*chunk:chunck_size*(chunk+1),:] if chunk!=chunks-1 else pts[chunck_size*chunk:,:]
        full_conf = np.zeros((full_pts.shape[0],n_frames))
        full_prob = np.zeros((full_pts.shape[0],n_frames,32))
        
        for i in range(Ts.shape[0]):
            
            data = np.load(os.path.join(prob_p,f'{img_list[i]}.npz'))
            probability = data['probability']
            confidence = data['confidence']
            
            pcd_current_frame = T_pcd(np.linalg.inv(Ts[i]), full_pts)
            pcd_canvas = ((K) @ pcd_current_frame.T).T
            pcd_canvas = np.round((pcd_canvas/pcd_canvas[:,-1][...,None])[:,:2]).astype(np.int16)
            
            # mask visible points
            mask = pcd_current_frame[:,-1] > 0
            mask = np.logical_and(mask, pcd_canvas[:,0] >= 0)
            mask = np.logical_and(mask , pcd_canvas[:,0] < W-0.5) #u
            mask = np.logical_and(mask, pcd_canvas[:,1] >= 0) #v
            mask = np.logical_and(mask, pcd_canvas[:,1] < H-0.5) #v
            diameter = 1
            camera = [0, 0, 0]
            radius = diameter * 5000
            pcd_cur_ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_current_frame[mask]))
            try:
                _, pt_map = pcd_cur_.hidden_point_removal(camera, radius)
            except:
                continue
            #pcd = pcd_cur_.select_by_index(pt_map)
            #o3d.visualization.draw_geometries([pcd])
            mask_ = np.zeros(pcd_current_frame[mask].shape[0])
            mask_[pt_map] = 1        
            # mask occluded points
            mask[mask] = mask_==1
            
            uv_coord = pcd_canvas[mask]
            #full_seg[mask,frame] = seg_m[uv_coord[:,1], uv_coord[:,0]]
            #
            full_prob[mask,i,:] = probability[uv_coord[:,1], uv_coord[:,0]]
            full_conf[mask,i] = confidence[uv_coord[:,1], uv_coord[:,0]]
            
        full_conf_ = np.sum(full_conf, axis=-1)[...,None]
        full_conf_[full_conf_==0] = 1
        norm_conf = full_conf/full_conf_ # 
        weighted_prob = np.sum(full_prob * norm_conf[...,None], axis=1)
        
        seg_all = np.argmax(weighted_prob,axis=-1)
        colors_all = reduced_colormap[seg_all.astype(int),:]
        
        seg_labels = np.concatenate((seg_labels, seg_all), axis=0)
        colors = np.vstack((colors, colors_all))
        
    pcd.colors = o3d.utility.Vector3dVector(colors)
    #if vis:
    #    cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #    o3d.visualization.draw_geometries([pcd, cf])
    return pcd, seg_labels

def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument(
        "--seq_name",
        default=' ',
        help="folder name to the output of droid slam for each video"
    )
    parser.add_argument(
        "--data_path",
        default='/home/selim/data/galaxy_note20',
        help="directory to where droid slam results are saved"
    )       
    parser.add_argument(
        "--path_panoptic",
        default='/home/junjee.chao/work/panoptic-lifting',
        help="path to panoptic lifting "
    )   
    parser.add_argument(
        "--path_m2f",
        default='/home/junjee.chao/work/Mask2Former/',
        help="path to Mask2Former"
    )   
    parser.add_argument(
        "--path_simplerecon",
        default='/home/junjee.chao/work/simplerecon/',
        help="path to Simplerecon"
    )
    return parser

def post_process_m2f_aug(args, seq_name):
    src_folder = Path(os.path.join(args.data_path,seq_name,'m2f','mask2former'))#Path('/home/jj/work/room_recon/data/redwood_apartment/mask2former/')
    rgb_folder_name = 'rgb'
    sc_classes='extended'

    coco_to_scannet = {}
    thing_semantics = get_thing_semantics(sc_classes)
    for cidx, cllist in enumerate([x.strip().split(',') for x in Path(f"{args.path_panoptic}/resources/scannet_{sc_classes}_to_coco.csv").read_text().strip().splitlines()]):
        for c in cllist[1:]:
            coco_to_scannet[c.split('/')[1]] = cidx + 1
    instance_ctr = 1
    instance_to_semantic = {}
    instance_ctr_notta = 1
    segment_ctr = 1
    instance_to_semantic_notta = {}
    (src_folder / "m2f_instance").mkdir(exist_ok=True)
    (src_folder / "m2f_semantics").mkdir(exist_ok=True)
    (src_folder / "m2f_notta_instance").mkdir(exist_ok=True)
    (src_folder / "m2f_notta_semantics").mkdir(exist_ok=True)
    (src_folder / "m2f_feats").mkdir(exist_ok=True)
    (src_folder / "m2f_probabilities").mkdir(exist_ok=True)
    (src_folder / "m2f_invalid").mkdir(exist_ok=True)
    (src_folder / "m2f_segments").mkdir(exist_ok=True)
    
    if not len(os.listdir(str((src_folder / "m2f_segments")))) == len(os.listdir(str((src_folder.parent.absolute() / rgb_folder_name)))):
        for idx, fpath in enumerate(sorted(list((src_folder.parent.absolute() / rgb_folder_name).iterdir()), key=lambda x: x.stem)):
            print(idx,fpath)
            data = torch.load(gzip.open(src_folder / f'{fpath.stem}.ptz'), map_location='cpu')
            probability, confidence, confidence_notta = data['probabilities'], data['confidences'], data['confidences_notta']
        
            semantic, instance, invalid_mask, instance_ctr, instance_to_semantic = convert_from_mask_to_semantics_and_instances_no_remap(data['mask'], data['segments'], coco_to_scannet, thing_semantics, instance_ctr, instance_to_semantic)
            semantic_notta, instance_notta, _, instance_ctr_notta, instance_to_semantic_notta = convert_from_mask_to_semantics_and_instances_no_remap(data['mask_notta'], data['segments_notta'], coco_to_scannet, thing_semantics,
                                                                                                                                                      instance_ctr_notta, instance_to_semantic_notta)
            segment_mask = torch.zeros_like(data['mask'])
            for s in data['segments']:
                segment_mask[data['mask'] == s['id']] = segment_ctr
                segment_ctr += 1
            Image.fromarray(segment_mask.numpy().astype(np.uint16)).save(src_folder / "m2f_segments" / f"{fpath.stem}.png")
            Image.fromarray(semantic.numpy().astype(np.uint16)).save(src_folder / "m2f_semantics" / f"{fpath.stem}.png")
            Image.fromarray(instance.numpy()).save(src_folder / "m2f_instance" / f"{fpath.stem}.png")
            Image.fromarray(semantic_notta.numpy().astype(np.uint16)).save(src_folder / "m2f_notta_semantics" / f"{fpath.stem}.png")
            Image.fromarray(instance_notta.numpy()).save(src_folder / "m2f_notta_instance" / f"{fpath.stem}.png")
            Image.fromarray(invalid_mask.numpy().astype(np.uint8) * 255).save(src_folder / "m2f_invalid" / f"{fpath.stem}.png")
            np.savez_compressed(src_folder / "m2f_probabilities" / f"{fpath.stem}.npz", probability=probability.float().numpy(), confidence=confidence.float().numpy(), confidence_notta=confidence_notta.float().numpy())
        

def run_m2f_aug_single_scene(args, seq_name):
    Ts = []
    m2f_base_p = os.path.join(args.data_path,seq_name,'m2f')
    os.makedirs(m2f_base_p,exist_ok=True)
    m2f_rgb_p = os.path.join(m2f_base_p,'rgb')
    os.makedirs(m2f_rgb_p,exist_ok=True)
    with open(f'{args.path_simplerecon}/data_splits/{seq_name}/test_eight_view_deepvmvs.txt') as f:
        inds = f.readlines()
    img_list = []
    for i in range(len(inds)):
        img_name = inds[i].split(' ')[1]
        img_list.append(img_name)
        Ts.append(np.loadtxt(f'{args.data_path}/{seq_name}/poses/{img_name}.txt'))
        shutil.copy(f'{args.data_path}/{seq_name}/images/{img_name}.png', m2f_rgb_p)
        
    # # step 2: run mask2former
    cwd = os.getcwd()
    os.chdir(f'{args.path_m2f}/demo')
    #subprocess.run(["cd","../Mask2Former/demo"])
    subprocess.run(["python",f"{args.path_m2f}/demo/demo.py","--config-file",
                    f"{args.path_m2f}/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
                    "--input", f'{m2f_rgb_p}',
                    "--opts", "MODEL.WEIGHTS", f"{args.path_m2f}/model_weights/model_final_47429163_0.pkl"
                    ])
    os.chdir(cwd)
    # # post process
    post_process_m2f_aug(args, seq_name)
    
    # fuse semantics
    pcd, seg_labels = fuse_m2f(args, seq_name, np.asarray(Ts), img_list, number_of_points=100000, chunck_size=30000, vis=False)
    o3d.io.write_point_cloud(f'{args.data_path}/{seq_name}/viola/{seq_name}_semantics.ply',pcd)
    data = {}
    data['semantic_labels'] = seg_labels
    data['pts'] = np.asarray(pcd.points)
    data['colors'] = np.asarray(pcd.colors)
    np.save(f'{args.data_path}/{seq_name}/viola/{seq_name}_semantics.npy',data)
    
if __name__ == "__main__":
    args = get_parser().parse_args()  
    
    # scene_names = ['arcore-dataset-2023-10-26-15-06-01','arcore-dataset-2023-10-27-16-01-41',
    #                'arcore-dataset-2023-10-27-16-06-46','arcore-dataset-2023-10-27-18-39-11',
    #                'arcore-dataset-2023-10-27-18-46-17','arcore-dataset-2023-11-01-16-35-47',
    #                'arcore-dataset-2023-11-01-17-24-02']

    scene_names = ['arcore-dataset-2023-11-02-14-16-35']
    
    for scene in scene_names:
        run_m2f_aug_single_scene(args,scene)
