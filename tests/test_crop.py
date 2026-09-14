# -*- coding: utf-8 -*-
#Copyright (C) 2026. Sean Moran. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""A batch size > 1 needs uniform image sizes; --crop_size provides them.

FiveK images vary in size, so the default DataLoader collation fails for
batch>1. Dataset(crop_size=N) random-crops training pairs to NxN so batches
collate. This test builds a tiny dataset from the bundled (variable-size)
example images and checks that crop_size yields uniform NxN batches.
"""
import glob
from PIL import Image
import os

import torch

from data import Adobe5kDataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "adobe5k_dpe")


def _tiny_dataset(tmp_path, crop_size):
    in_dir = os.path.join(EX, "deeplpf_example_test_input")
    out_dir = os.path.join(EX, "deeplpf_example_test_output")
    ins = {os.path.basename(f).split("-")[0] for f in glob.glob(in_dir + "/*.png")}
    outs = {os.path.basename(f).split("-")[0] for f in glob.glob(out_dir + "/*.png")}
    ids = sorted(ins & outs)
    id_file = tmp_path / "ids.txt"
    id_file.write_text("\n".join(ids) + "\n")
    data_dict = Adobe5kDataLoader(data_dirpath=EX + "/", img_ids_filepath=str(id_file)).load_data()
    return Dataset(data_dict=data_dict, normaliser=255, is_valid=False, crop_size=crop_size), len(ids)


def test_crop_size_gives_uniform_batches(tmp_path):
    ds, n = _tiny_dataset(tmp_path, crop_size=256)
    assert n >= 3
    dl = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    assert tuple(batch["input_img"].shape) == (3, 3, 256, 256)
    assert tuple(batch["output_img"].shape) == (3, 3, 256, 256)


def test_examples_really_are_mixed_size():
    """The premise behind --crop_size: the bundled pairs are not one shape.

    The export fixes the long edge, not both, so the set holds 512x341, 512x343
    and one portrait 341x512.
    """
    sizes = {Image.open(f).size
             for f in glob.glob(os.path.join(EX, "deeplpf_example_test_input", "*.png"))}
    assert len(sizes) > 1, "expected mixed sizes, got %s" % sizes


def test_crop_size_makes_a_batch_collate(tmp_path):
    """With a crop, images of differing sizes stack into one batch.

    This is the invariant --crop_size exists to provide. Asserting instead that
    the uncropped case *fails* would be asserting a detail of the DataLoader's
    collate, which is torch's to change and has.
    """
    ds, _ = _tiny_dataset(tmp_path, crop_size=128)
    dl = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    assert batch["input_img"].shape == (3, 3, 128, 128)
    assert batch["output_img"].shape == (3, 3, 128, 128)
