import argparse
import numpy as np
import os
import pandas as pd
import torch


def norm_intensity(intensity):
    
    bottom, up = np.percentile(intensity, 1), np.percentile(intensity, 99)
    intensity[intensity < bottom] = bottom
    intensity[intensity > up] = up
    intensity -= bottom
    intensity = intensity / (up - bottom)
    
    return intensity


def main_process():
    
    training_stations = ['1-1', '1-2', '1-3', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '1-11', '1-13', '1-14', '1-16', '1-17', '2-1', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-11', '2-12', '2-13']
    validation_stations = ['1-4', '1-12', '1-15', '2-2', '2-10', '2-14']
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    
    files = os.listdir(args.dataset_root)
    
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)
    if not os.path.exists(os.path.join(args.output_root, 'training_set')):
        os.makedirs(os.path.join(args.output_root, 'training_set'))
    if not os.path.exists(os.path.join(args.output_root, 'validation_set')):
        os.makedirs(os.path.join(args.output_root, 'validation_set'))
    
    stations = {}
    for file in files:
        station = file.rsplit('-', 1)[0]
        if station not in training_stations and station not in validation_stations:
            continue
        if station not in stations.keys():
            stations[station] = []
        stations[station].append(file)
    
    for station in stations.keys():
        pc = []
        for i in range(len(stations[station])):
            ring = pd.read_csv(os.path.join(args.dataset_root, stations[station][i]), sep=' ', header=None)
            ring = np.asarray(ring)
            pc.append(ring)
        pc = np.vstack(pc)
    
        pc[:, 3] = norm_intensity(pc[:, 3])
        pc[:, 0:3] -= np.mean(pc[:, 0:3], axis=0)
        np.random.shuffle(pc)
        
        coords = pc[:, 0:3]
        colors = pc[:, 3].reshape(-1, 1).repeat(3, axis=1) * 255
        semantic_gt = pc[:, -1].reshape(-1, 1)
        semantic_gt = np.asarray(semantic_gt, dtype=int)
        
        save_dict = dict(coord=coords, color=colors, semantic_gt=semantic_gt)
        
        if station in training_stations:
            save_path = os.path.join(args.output_root, 'training_set', station + '.pth')
        else:
            save_path = os.path.join(args.output_root, 'validation_set', station + '.pth')
        torch.save(save_dict, save_path)


if __name__ == "__main__":
    
    main_process()
