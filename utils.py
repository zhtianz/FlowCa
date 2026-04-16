"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is from: [https://github.com/3dem/model-angelo/blob/main/model_angelo/utils/save_pdb_utils.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""

import importlib.util
import torch.nn as nn
import sys
import os
from typing import List
import argparse
from collections import namedtuple
from typing import List, Tuple

import einops
import mrcfile as mrc
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve
import sys
import os
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.StructureBuilder import StructureBuilder
from copy import deepcopy

def get_model_from_file(file_path: str) -> nn.Module:
    spec = importlib.util.spec_from_file_location("network", file_path)
    network = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(network)
    return network.Model()

def filter_useless_warnings():
    import warnings

    warnings.filterwarnings("ignore", ".*nn\.functional\.upsample is deprecated.*")
    warnings.filterwarnings("ignore", ".*none of the inputs have requires_grad.*")
    warnings.filterwarnings("ignore", ".*with given element none.*")
    warnings.filterwarnings("ignore", ".*invalid value encountered in true\_divide.*")

def is_relion_abort(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, "RELION_JOB_ABORT_NOW"))


def write_relion_job_exit_status(
    directory: str, status: str, pipeline_control: str = "",
):
    if pipeline_control != "":
        open(os.path.join(directory, f"RELION_JOB_EXIT_{status}"), "a").close()
    elif status == "FAILURE":
        sys.exit(1)


def abort_if_relion_abort(directory: str):
    if is_relion_abort(directory):
        write_relion_job_exit_status(directory, "ABORTED")
        print("Aborting now...")
        sys.exit(1)

def get_device_name(device_name: str) -> str:
    if device_name is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_name == "cpu":
        return "cpu"
    if device_name.startswith("cuda:"):
        return device_name
    if device_name.isnumeric():
        return f"cuda:{device_name}"
    else:
        raise RuntimeError(
            f"Device name: {device_name} not recognized. "
            f"Either do not set, set to cpu, or give a number"
        )

def get_device_names(device_name_str: str) -> List[str]:
    if device_name_str is None or "," not in device_name_str:
        return [get_device_name(device_name_str)]
    else:
        return [get_device_name(x.strip()) for x in device_name_str.split(",") if len(x.strip()) > 0]


def extend_edge(binary_mask, edge, kernel, ramp):
    smooth_mask = np.copy(binary_mask)
    prev_mask = np.copy(binary_mask).astype(bool)
    for i in range(edge):
        mask = convolve(prev_mask.astype(np.float32), kernel) > 0
        skin = mask & ~prev_mask
        prev_mask = mask
        smooth_mask[skin] = ramp[i]
    return smooth_mask


def get_mask_from_grid(
    grid: np.ndarray,
    ini_threshold: float = 0.01,
    extend_inimask: int = 3,
    width_soft_edge: int = 3,
) -> np.ndarray:
    binary_mask = np.zeros_like(grid, dtype=np.float32)
    binary_mask[grid > ini_threshold] = 1

    kernel = np.zeros((3, 3, 3))
    kernel[:, 1, 1] = 1
    kernel[1, :, 1] = 1
    kernel[1, 1, :] = 1

    if extend_inimask > 0:
        binary_mask = extend_edge(
            binary_mask, extend_inimask, kernel, np.ones((extend_inimask,)),
        )

    if width_soft_edge > 0:
        binary_mask = extend_edge(
            binary_mask,
            width_soft_edge,
            kernel,
            ramp=np.cos(np.linspace(0, np.pi, width_soft_edge)) * 0.5 + 0.5,
        )

    return binary_mask

def apply_lowpass_filter_to_map(
    grid: np.ndarray,
    voxel_size: float,
    lowpass_ang: float,
    filter_edge_width: int = 2,
    use_cosine_kernel: bool = True,
) -> np.ndarray:
    grid_ft = np.fft.rfftn(np.fft.fftshift(grid))
    spectral_radius = get_fourier_shells(grid_ft)
    ori_size = grid.shape[0]

    ires_filter = round((ori_size * voxel_size) / lowpass_ang)
    filter_edge_halfwidth = filter_edge_width // 2

    edge_low = max(0, (ires_filter - filter_edge_halfwidth) / ori_size)
    edge_high = min(grid_ft.shape[0], (ires_filter + filter_edge_halfwidth) / ori_size)
    edge_width = edge_high - edge_low

    res = spectral_radius / ori_size
    scale_spectrum = np.zeros_like(res)
    scale_spectrum[res < edge_low] = 1

    if use_cosine_kernel:
        scale_spectrum[(res >= edge_low) & (res <= edge_high)] = 0.5 + 0.5 * np.cos(
            np.pi
            * (res[(res >= edge_low) & (res <= edge_high)] - edge_low)
            / edge_width
        )

    grid_ft *= scale_spectrum
    grid = np.fft.ifftshift(np.fft.irfftn(grid_ft))
    return grid
def get_spherical_mask(grid: np.ndarray) -> np.ndarray:
    ls = np.linspace(-grid.shape[0] // 2, grid.shape[0] // 2, grid.shape[0])
    r = np.stack(np.meshgrid(ls, ls, ls, indexing="ij"), -1)
    r = np.linalg.norm(r, ord=2, axis=-1)
    mask = np.zeros_like(grid, dtype=bool)
    mask[r < (grid.shape[0] / 2 + 1)] = True
    return mask

def get_auto_mask(grid: np.ndarray, voxel_size: float) -> np.ndarray:
    lowpass_grid = apply_lowpass_filter_to_map(grid, voxel_size, 15)
    s_mask = get_spherical_mask(lowpass_grid)
    lowpass_grid[~s_mask] = 0

    threshold = np.quantile(lowpass_grid[lowpass_grid > 0], q=0.97)
    extend_mask_value = max(5, round(0.01 * grid.shape[0]) + 1)
    mask_grid = get_mask_from_grid(
        lowpass_grid,
        threshold,
        extend_inimask=extend_mask_value,
        width_soft_edge=extend_mask_value,
    )
    return mask_grid




def get_fourier_shells(f):
    (z, y, x) = f.shape
    Z, Y, X = np.meshgrid(
        np.linspace(-z // 2, z // 2 - 1, z),
        np.linspace(-y // 2, y // 2 - 1, y),
        np.linspace(0, x - 1, x),
        indexing="ij",
    )
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    R = np.fft.ifftshift(R, axes=(0, 1))
    return R

def apply_bfactor_to_map(
    grid: np.ndarray, voxel_size: float, bfactor: float
) -> np.ndarray:
    grid_ft = np.fft.rfftn(np.fft.fftshift(grid))
    spectral_radius = get_fourier_shells(grid_ft)
    ori_size = grid.shape[0]

    res = spectral_radius / (ori_size * voxel_size)
    scale_spectrum = np.exp(-bfactor / 4 * np.square(res))
    grid_ft *= scale_spectrum

    grid = np.fft.ifftshift(np.fft.irfftn(grid_ft))
    return grid


MRCObject = namedtuple("MRCObject", ["grid", "voxel_size", "global_origin"])

def load_mrc(mrc_fn: str, multiply_global_origin: bool = True) -> MRCObject:
    mrc_file = mrc.open(mrc_fn, "r")
    voxel_size = float(mrc_file.voxel_size.x)

    if voxel_size <= 0:
        raise RuntimeError(f"Seems like the MRC file: {mrc_fn} does not have a header.")

    c = mrc_file.header["mapc"]
    r = mrc_file.header["mapr"]
    s = mrc_file.header["maps"]

    global_origin = mrc_file.header["origin"]
    global_origin = np.array([global_origin.x, global_origin.y, global_origin.z])
    nstart = np.array([mrc_file.header["nxstart"],mrc_file.header["nystart"],mrc_file.header["nzstart"]])
    print("global_origin CryFold ",global_origin,nstart)
    temp1 = [c - 1, r - 1, s - 1]
    temp_start = np.zeros(3)
    for temp_index in range(3):
        temp_start[temp1[temp_index]] = nstart[temp_index]
    global_origin = global_origin + temp_start
    print("global_origin CryFold ",global_origin,temp_start)
    ####
    if multiply_global_origin:
        global_origin *= mrc_file.voxel_size.x

    if c == 1 and r == 2 and s == 3:
        grid = mrc_file.data
    elif c == 3 and r == 2 and s == 1:
        grid = np.moveaxis(mrc_file.data, [0, 1, 2], [2, 1, 0])
    elif c == 3 and r == 1 and s == 2:
       #grid = np.moveaxis(mrc_file.data, [0, 2, 1], [2, 1, 0])
        grid = np.moveaxis(mrc_file.data, [1, 0, 2], [2, 1, 0]) 
    elif c == 2 and r == 1 and s == 3:
        grid = np.moveaxis(mrc_file.data, [1, 2, 0], [2, 1, 0])
    elif c == 1 and r == 3 and s == 2:
        grid = np.moveaxis(mrc_file.data, [2, 0, 1], [2, 1, 0])
    elif c == 2 and r == 3 and s == 1:
        grid = np.moveaxis(mrc_file.data, [0, 2, 1], [2, 1, 0])
    else:
        raise RuntimeError("MRC file axis arrangement not supported!")
    

    return MRCObject(grid, voxel_size, global_origin)


def make_cubic(box):
    bz = np.array(box.shape)
    s = np.max(box.shape)
    s = max(s, 128)
    s += s % 2
    if np.all(box.shape == s):
        return box, np.zeros(3, dtype=np.int64), bz
    nbox = np.zeros((s, s, s))
    c = np.array(nbox.shape) // 2 - bz // 2
    nbox[c[0] : c[0] + bz[0], c[1] : c[1] + bz[1], c[2] : c[2] + bz[2]] = box
    return nbox, c, c + bz
def rescale_fourier(box, out_sz):
    if out_sz % 2 != 0:
        raise Exception("Bad output size")
    if box.shape[0] != box.shape[1] or box.shape[1] != (box.shape[2] - 1) * 2:
        raise Exception("Input must be cubic")

    ibox = np.fft.ifftshift(box, axes=(0, 1))
    obox = np.zeros((out_sz, out_sz, out_sz // 2 + 1), dtype=box.dtype)

    si = np.array(ibox.shape) // 2
    so = np.array(obox.shape) // 2

    if so[0] < si[0]:
        obox = ibox[
            si[0] - so[0] : si[0] + so[0],
            si[1] - so[1] : si[1] + so[1],
            : obox.shape[2],
        ]
    elif so[0] > si[0]:
        obox[
            so[0] - si[0] : so[0] + si[0],
            so[1] - si[1] : so[1] + si[1],
            : ibox.shape[2],
        ] = ibox
    else:
        obox = ibox

    obox = np.fft.ifftshift(obox, axes=(0, 1))

    return obox


def rescale_real(box, out_sz):
    if out_sz != box.shape[0]:
        f = np.fft.rfftn(box)
        f = rescale_fourier(f, out_sz)
        box = np.fft.irfftn(f)

    return box

def normalize_voxel_size(density, in_voxel_sz, target_voxel_size=1.0, is_mask=False):
    (iz, iy, ix) = np.shape(density)

    assert iz % 2 == 0 and iy % 2 == 0 and ix % 2 == 0
    assert ix == iy == iz

    in_sz = ix
    out_sz = int(round(in_sz * in_voxel_sz / target_voxel_size))
    if out_sz % 2 != 0:
        vs1 = in_voxel_sz * in_sz / (out_sz + 1)
        vs2 = in_voxel_sz * in_sz / (out_sz - 1)
        if np.abs(vs1 - target_voxel_size) < np.abs(vs2 - target_voxel_size):
            out_sz += 1
        else:
            out_sz -= 1

    out_voxel_sz = in_voxel_sz * in_sz / out_sz
    if is_mask:
       
        from scipy.ndimage import zoom
        zoom_factor = out_sz / in_sz
        density = zoom(density, zoom_factor, order=0)
    else:
        density = rescale_real(density, out_sz)

    return density, out_voxel_sz


def make_model_angelo_grid(grid, voxel_size, global_origin, target_voxel_size=1.5, is_mask=False):
    grid, shift, _ = make_cubic(grid)
    global_origin[0] -= shift[2] * voxel_size
    global_origin[1] -= shift[1] * voxel_size
    global_origin[2] -= shift[0] * voxel_size

    grid, voxel_size = normalize_voxel_size(
        grid, voxel_size, target_voxel_size=target_voxel_size, is_mask=is_mask
    )
    return MRCObject(grid, voxel_size, global_origin)

def get_lattice_meshgrid_np(shape, no_shift=False):
    linspace = np.linspace(
        0.5 if not no_shift else 0, shape - (0.5 if not no_shift else 1), shape,
    )
    mesh = np.stack(np.meshgrid(linspace, linspace, linspace, indexing="ij"), axis=-1,)
    return mesh

def get_local_std(grid: torch.Tensor, kernel_size: int = 10) -> torch.Tensor:
    assert len(grid.shape) == 5
    grid_mean = grid.clone()
    grid_squared = grid.square()
    kernel = torch.exp(-torch.linspace(-1.5, 1.5, 2 * kernel_size + 1).square()).to(
        grid.device
    )
    kernel = kernel[None, None, :, None, None] / kernel.sum()
    for i in range(3):
        grid_mean = einops.rearrange(grid_mean, "b c x y z -> b c z x y")
        grid_mean = F.conv3d(grid_mean, kernel, padding="same")
        grid_squared = einops.rearrange(grid_squared, "b c x y z -> b c z x y")
        grid_squared = F.conv3d(grid_squared, kernel, padding="same")
    return grid_squared.sub_(grid_mean.square_()).relu_().sqrt_()

# TODO: Change save_mrc API to accept an MRCObject and a filename only
# From now on, grids should be accompanied inside an MRCObject so that the global_origin
# and voxel_sizes come too
def save_mrc(grid, voxel_size, origin, filename):
    (z, y, x) = grid.shape
    o = mrc.new(filename, overwrite=True)
    o.header["cella"].x = x * voxel_size
    o.header["cella"].y = y * voxel_size
    o.header["cella"].z = z * voxel_size
    o.header["origin"].x = origin[0]
    o.header["origin"].y = origin[1]
    o.header["origin"].z = origin[2]
    out_box = np.reshape(grid, (z, y, x))
    o.set_data(out_box.astype(np.float32))
    o.update_header_stats()
    o.flush()
    o.close()



def is_relion_abort(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, "RELION_JOB_ABORT_NOW"))


def write_relion_job_exit_status(
    directory: str, status: str, pipeline_control: str = "",
):
    if pipeline_control != "":
        open(os.path.join(directory, f"RELION_JOB_EXIT_{status}"), "a").close()
    elif status == "FAILURE":
        sys.exit(1)


def abort_if_relion_abort(directory: str):
    if is_relion_abort(directory):
        write_relion_job_exit_status(directory, "ABORTED")
        print("Aborting now...")
        sys.exit(1)


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


def points_to_pdb(path_to_save, points):
    struct = StructureBuilder()
    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")
    struct.init_chain("1")
    for i, point in enumerate(points):
        struct.set_line_counter(i)
        struct.init_residue(f"ALA", " ", i, " ")
        struct.init_atom("CA", point, 0, 1, " ", "CA", "C")
    struct = struct.get_structure()
    save_structure_to_cif(struct, path_to_save)


def ca_ps_to_pdb(path_to_save, ca_points, p_points):
    struct = StructureBuilder()
    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")
    struct.init_chain("1")
    ca_num = len(ca_points)
    for i, point in enumerate(ca_points):
        struct.set_line_counter(i)
        struct.init_residue(f"ALA", " ", i, " ")
        struct.init_atom(
            name="CA",
            coord=point,
            b_factor=0,
            occupancy=1,
            altloc=" ",
            fullname="CA",
            element="C",
        )
    struct.init_chain("2")
    for i, point in enumerate(p_points):
        struct.set_line_counter(ca_num + i)
        struct.init_residue(f"A", " ", ca_num + i, " ")
        struct.init_atom(
            name="P",
            coord=point,
            b_factor=0,
            occupancy=1,
            altloc=" ",
            fullname="P",
            element="P",
        )
    struct = struct.get_structure()
    save_structure_to_cif(struct, path_to_save)






