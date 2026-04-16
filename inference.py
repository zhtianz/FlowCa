import argparse
import mrcfile
from copy import deepcopy
import os
import torch
import numpy as np
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.StructureBuilder import StructureBuilder
from scipy.spatial import cKDTree
import math
from itertools import product
from random import shuffle
import tqdm
import sys

from utils import (
    abort_if_relion_abort, 
    get_device_names,
    get_lattice_meshgrid_np,
    load_mrc,
    make_model_angelo_grid,
    save_mrc,
)
from multi_gpu_wrapper import MultiGPUWrapper
import tempfile
import warnings
from FlowModel import DensityFlowMatching as DensityFlowMatching

seed = 42  # 可以是任何你喜欢的整数
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_args_parser():
    parser = argparse.ArgumentParser('FlowCα: Accurate Cα Atom Prediction in cryo-EM Maps via Flow Matching', add_help=True)
    parser.add_argument('--device', default='cuda:0', 
                        help='Device to use for inference')
    parser.add_argument('--batch_size', default=24, type=int, 
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model
    parser.add_argument('--input_size', default=64, type=int, help='Input patch size (pixels) for density map crops')
    parser.add_argument('--model_path', default="checkpoint.pth", help='Pretrained FlowCα model checkpoint path')
    # Data
    parser.add_argument('--map_path', help='Path to input density map file (mrc/map format)')
    parser.add_argument("--contour", type=float, help="Recommended contour level for density map")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for Cα prediction")
    return parser


def standardize_map():
    
    if not args.map_path.endswith("mrc") and not args.map_path.endswith("map") :
        warnings.warn(f"The file {args.map_path} does not end with '.mrc' or 'map'\nPlease make sure it is an MRC file.")
    grid_np, voxel_size, global_origin = load_mrc(args.map_path)   
    
    ca_mask = np.ones_like(grid_np)
    threshold = args.contour * 0.8
    ca_mask[grid_np < threshold] = 0

    ca_mask, voxel_size_temp, global_origin_temp = make_model_angelo_grid(
        np.copy(ca_mask), voxel_size, global_origin.copy(), target_voxel_size=1.5, is_mask=True
    )
    
    ca_mask = (ca_mask > 0.5).astype(np.float32)
    
    grid_np, voxel_size, global_origin = make_model_angelo_grid(   
        np.copy(grid_np), voxel_size, global_origin, target_voxel_size=1.5, is_mask=False
    )
    
    mask_grid = np.ones_like(grid_np)
    grid_np = grid_np * mask_grid
    mask_grid = mask_grid.astype(np.float32)
    grid_np = (grid_np - np.mean(grid_np)) / (np.std(grid_np) + 1e-6)    

    return grid_np, voxel_size, global_origin, mask_grid, ca_mask

def infer(grid_np, model, state_dict_path, output_dir, device_names, mask_grid):
    bz = 64       
    stride  = 16
    crop = 6 
    num_class = 1

    grid = torch.Tensor(grid_np)    
    output = torch.zeros(num_class, *grid.shape[-3:])
    count_output = torch.zeros(*grid.shape[-3:])

    x_coordinates = [i * stride for i in range((grid.shape[-1] - bz) // stride)] + [
        grid.shape[-1] - 1 - bz
    ]
    y_coordinates = [i * stride for i in range((grid.shape[-1] - bz) // stride)] + [
        grid.shape[-1] - 1 - bz
    ]
    z_coordinates = [i * stride for i in range((grid.shape[-1] - bz) // stride)] + [
        grid.shape[-1] - 1 - bz
    ]
    coordinates_to_infer = list(product(x_coordinates, y_coordinates, z_coordinates))
    shuffle(coordinates_to_infer)

    i = 0

    pbar = tqdm.tqdm(
        total=len(coordinates_to_infer), file=sys.stdout, position=0, leave=True,
    )
    with MultiGPUWrapper(model, state_dict_path, device_names) as wrapper:
        while i < len(coordinates_to_infer):
            meta_batch_list = []
            meta_batch_coordinates = []
            for _ in device_names:
                batch_grid = []
                batch_coordinates = []
                for _ in range(args.batch_size):
                    if i < len(coordinates_to_infer):
                        curr_coordinate = coordinates_to_infer[i]
                        coordinate_slice = np.s_[
                            ...,
                            curr_coordinate[0] : curr_coordinate[0] + bz,
                            curr_coordinate[1] : curr_coordinate[1] + bz,
                            curr_coordinate[2] : curr_coordinate[2] + bz,
                        ]
                        sliced_grid = grid[coordinate_slice][None].clone()
                        batch_coordinates.append(curr_coordinate)
                        batch_grid.append(torch.Tensor(sliced_grid))
                        i += 1
                if len(batch_grid) > 0:
                    batch_grid = torch.stack(batch_grid)          
                    meta_batch_list.append(batch_grid)            
                    meta_batch_coordinates.append(batch_coordinates)
            if len(meta_batch_list) > 0:
                meta_net_output = wrapper(meta_batch_list)
                pbar.update(sum(len(batch_grid) for batch_grid in meta_batch_list))
            abort_if_relion_abort(output_dir)

            for batch_coordinates, net_output in zip(meta_batch_coordinates, meta_net_output):
                for j, c in enumerate(batch_coordinates):
                    batch_slice = np.s_[
                        ...,
                        c[0] + crop : c[0] + bz - crop,
                        c[1] + crop : c[1] + bz - crop,
                        c[2] + crop : c[2] + bz - crop,
                    ]
                    net_output_batch = net_output[
                        j, ..., crop : bz - crop, crop : bz - crop, crop : bz - crop,
                    ]
                    output[batch_slice] += net_output_batch.cpu() 
                    count_output[batch_slice] += 1

    pbar.close()


    output = output.cpu().float().numpy()
    count_output = count_output.cpu().float().numpy() + 1e-6
    output = output / count_output
    output = output * mask_grid[None]

    return output

def cluster_kdtree(points, probs, neighbour_distance_threshold, prune_distance):
    for _ in range(3):
        kdtree = cKDTree(np.copy(points))
        n = 0

        new_points = np.copy(points)
        for p in points:
            neighbours = kdtree.query_ball_point(p, prune_distance)
            selection = list(neighbours)
            if len(neighbours) > 1 and np.sum(probs[selection]) > 0:
                keep_idx = np.argmax(probs[selection])
                prob_sum = np.sum(probs[selection])

                new_points[selection[keep_idx]] = (
                    np.sum(probs[selection][..., None] * points[selection], axis=0)
                    / prob_sum
                )
                probs[selection] = 0
                probs[selection[keep_idx]] = prob_sum

            n += 1

        points = new_points[probs > 0].reshape(-1, 3)
        probs = probs[probs > 0]
    
    kdtree = cKDTree(np.copy(points))
    for point_idx, point in enumerate(points):
        d, _ = kdtree.query(point, 2)
        if d[1] > neighbour_distance_threshold:
            points[point_idx] = np.nan

    points = points[~np.isnan(points).any(axis=-1)].reshape(-1, 3)
    return points

def get_lattice_meshgrid_np(grid, no_shift=False):
    d, h, w = grid.shape[-3], grid.shape[-2], grid.shape[-1]
    dlinspace = np.linspace(
        0.5 if not no_shift else 0, d - (0.5 if not no_shift else 1), d,
    )
    hlinspace = np.linspace(
        0.5 if not no_shift else 0, h - (0.5 if not no_shift else 1), h,
    )
    wlinspace = np.linspace(
        0.5 if not no_shift else 0, w - (0.5 if not no_shift else 1), w,
    )
    mesh = np.stack(np.meshgrid(dlinspace, hlinspace, wlinspace, indexing="ij"), axis=-1,)
    return mesh

def grid_to_points(
    grid, threshold, neighbour_distance_threshold, prune_distance=1.1,
):
    lattice = np.flip(get_lattice_meshgrid_np(grid, no_shift=True), -1)

    output_points_before_pruning = np.copy(lattice[grid > threshold, :].reshape(-1, 3))

    points = lattice[grid > threshold, :].reshape(-1, 3)
    probs = grid[grid > threshold]
    print(len(points))

    points = cluster_kdtree(points, probs, neighbour_distance_threshold, prune_distance)
    output_points = points
    print("output_points",output_points.shape)
    print("output_points_before_pruning",output_points_before_pruning.shape)
    return output_points, output_points_before_pruning


class ModelAngeloMMCIFIO(MMCIFIO):
    def _save_dict(self, out_file):
        label_seq_id = deepcopy(self.dic["_atom_site.auth_seq_id"])
        auth_seq_id = deepcopy(self.dic["_atom_site.auth_seq_id"])
        self.dic["_atom_site.label_seq_id"] = label_seq_id
        self.dic["_atom_site.auth_seq_id"] = auth_seq_id
        return super()._save_dict(out_file)
def save_structure_to_cif(structure, path_to_save: str):
    io = ModelAngeloMMCIFIO()
    io.set_structure(structure)
    io.save(path_to_save)

def points_to_pdb(path_to_save, points, origin=(0,0,0)):
    struct = StructureBuilder()
    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")
    struct.init_chain("1")
    for i, point in enumerate(points):
        adjusted_point = (point[0] + origin[0], point[1] + origin[1], point[2] + origin[2])
        struct.set_line_counter(i)
        struct.init_residue(f"ALA", " ", i, " ")
        struct.init_atom("CA", adjusted_point, 0, 1, " ", "CA", "C")
    struct = struct.get_structure()
    save_structure_to_cif(struct, path_to_save)



def main(args):

    state_dict_path = args.model_path

    device_names = get_device_names(args.device)
    print("device_names ",device_names)

    
    device = torch.device(device_names[0])  
    print("Using device: ", device, args.model_path)
    
    checkpoint = torch.load(args.model_path, map_location=device)

    print(f" infer 检查点键: {list(checkpoint.keys())}")
    config = checkpoint['config']
    
    model = DensityFlowMatching(config)	
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint['model_state_dict'] 
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 去除 'module.'
            else:
                new_key = key
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict) 
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    input_path = args.map_path
    input_dir, input_filename = os.path.split(input_path)

    grid_np, voxel_size, global_origin, mask_grid, ca_mask = standardize_map()
    
    output = infer(grid_np, model, state_dict_path, input_dir, device_names, mask_grid)
    ca_grid = output[0]
    
    ca_grid = ca_grid * ca_mask

    ca_max = ca_grid.max()
    if ca_max < args.threshold:
        args.threshold = ca_max * 0.4
    
    output_ca_points, output_ca_points_before_pruning = grid_to_points(
        ca_grid, args.threshold, 6 / voxel_size
    )
   
    points_to_pdb(
        os.path.join(input_dir, "output_ca_points_before_pruning.cif"),
        voxel_size * output_ca_points_before_pruning, global_origin
    )
    output_file_path = os.path.join(input_dir, "see_alpha_output_ca.cif")
    points_to_pdb(
        output_file_path, voxel_size * output_ca_points, global_origin 
    )
   
import time   
if __name__ == '__main__':

    if len(sys.argv) == 1:
        parser = get_args_parser()
        parser.print_help()
        sys.exit(0)

    start_time = time.time()
    args = get_args_parser()
    args = args.parse_args()
    main(args)

    end_time = time.time()
    runtime_seconds = end_time - start_time
    runtime_minutes = runtime_seconds / 60
    print(f"inference_final2.py: Run time {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")


