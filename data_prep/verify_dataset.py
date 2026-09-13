# -*- coding: utf-8 -*-
#Copyright (C) 2026. Sean Moran. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
#!/usr/bin/env python
"""Verify a prepared Adobe-DPE dataset before training.

Runs the repo's own Adobe5kDataLoader over a prepared data directory for each
split file and reports how many image ids were paired (input + output found),
which are missing, and whether a sample loads as a 3-channel image. If the
repo's bundled example images are available it also compares your exported
inputs against them. Those examples are themselves an export made with the
recipe in docs/ADOBE_DPE_DATASET.md, so the comparison answers "does my
Lightroom render these ten the way yours did" - a check on the develop
settings, not evidence about the 2020 release.

Expected layout of DATA_DIR (see docs/ADOBE_DPE_DATASET.md):

    DATA_DIR/
      input/   a0001-....png, a0002-....png, ...   (Lightroom Input export)
      output/  a0001-....png, a0002-....png, ...   (Lightroom Expert C export)

Usage:
    python data_prep/verify_dataset.py DATA_DIR [--splits-dir adobe5k_dpe]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

# Make the repo modules importable regardless of where this is run from.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data import Adobe5kDataLoader  # noqa: E402


def check_split(data_dir, split_file):
    loader = Adobe5kDataLoader(data_dirpath=data_dir + "/", img_ids_filepath=split_file)
    try:
        # This script exists to diagnose an incomplete or misplaced export, so
        # the loader's own complaints about missing files are the condition it
        # is reporting on, not a reason to stop.
        data_dict = loader.load_data(require_output=False)
    except FileNotFoundError as exc:
        print(f"         {exc}")
        data_dict = {}

    with open(split_file) as f:
        wanted = [line.strip() for line in f if line.strip()]

    paired, missing_input, missing_output = [], [], []
    found_ids = set()
    for entry in data_dict.values():
        iid = os.path.basename(entry["input_img"]).split("-")[0] if entry.get("input_img") else None
        oid = os.path.basename(entry["output_img"]).split("-")[0] if entry.get("output_img") else None
        iid = iid or oid
        found_ids.add(iid)
        if entry.get("input_img") and entry.get("output_img"):
            paired.append(iid)
        elif not entry.get("input_img"):
            missing_input.append(iid)
        else:
            missing_output.append(iid)

    never_found = [w for w in wanted if w not in found_ids]
    return wanted, paired, missing_input, missing_output, never_found, data_dict


def sample_channels(data_dict):
    for entry in data_dict.values():
        if entry.get("input_img"):
            arr = np.array(Image.open(entry["input_img"]))
            return entry["input_img"], arr.shape
    return None, None


def compare_to_examples(data_dir):
    """Compare user exports against the repo's bundled example inputs.

    The bundled examples were exported with this repo's own recipe, so a small
    difference means your Lightroom develop settings agree with ours.
    """
    ex_in = os.path.join(REPO, "adobe5k_dpe", "deeplpf_example_test_input")
    if not os.path.isdir(ex_in):
        return
    examples = {}
    for fn in os.listdir(ex_in):
        if fn.lower().endswith((".png", ".jpg", ".tif")):
            examples.setdefault(fn.split("-")[0], fn)
    if not examples:
        return
    print("\n== comparison against bundled example inputs (does your Lightroom "
          "match ours?) ==")
    user_inputs = {}
    in_dir = os.path.join(data_dir, "input")
    if os.path.isdir(in_dir):
        for fn in os.listdir(in_dir):
            user_inputs.setdefault(fn.split("-")[0], os.path.join(in_dir, fn))
    for iid, ex_fn in sorted(examples.items()):
        if iid not in user_inputs:
            print(f"  {iid}: not present in your input/ folder (skip)")
            continue
        ex = np.array(Image.open(os.path.join(ex_in, ex_fn)).convert("RGB")).astype(np.float32)
        us = Image.open(user_inputs[iid]).convert("RGB").resize((ex.shape[1], ex.shape[0]))
        us = np.array(us).astype(np.float32)
        mad = np.abs(ex - us).mean()
        # Heuristic: resampling/format differences alone give a small mean
        # difference; a large one suggests the input develop settings differ.
        verdict = "close (preprocessing looks consistent)" if mad < 15 else "LARGE DIFF (check input develop settings)"
        print(f"  {iid}: mean|Δ| = {mad:6.2f} / 255  -> {verdict}")


def main():
    ap = argparse.ArgumentParser(description="Verify a prepared Adobe-DPE dataset.")
    ap.add_argument("data_dir", help="directory containing input/ and output/ subfolders")
    ap.add_argument("--splits-dir", default=os.path.join(REPO, "adobe5k_dpe"),
                    help="directory containing images_{train,valid,test}.txt")
    args = ap.parse_args()

    ok = True
    for split in ("train", "valid", "test"):
        split_file = os.path.join(args.splits_dir, f"images_{split}.txt")
        if not os.path.exists(split_file):
            print(f"[{split}] split file missing: {split_file}")
            continue
        wanted, paired, mi, mo, nf, dd = check_split(args.data_dir, split_file)
        print(f"[{split:5s}] wanted {len(wanted):4d} | paired {len(paired):4d} | "
              f"missing-input {len(mi)} | missing-output {len(mo)} | not-found {len(nf)}")
        if len(paired) != len(wanted):
            ok = False
            for label, ids in (("missing-input", mi), ("missing-output", mo), ("not-found", nf)):
                if ids:
                    print(f"         {label}: {ids[:10]}{' ...' if len(ids) > 10 else ''}")
        path, shape = sample_channels(dd)
        if shape is not None:
            chan = shape[2] if len(shape) == 3 else 1
            print(f"         sample {os.path.basename(path)} shape={shape} ({chan} channels)")
            if chan not in (3, 4):
                print("         WARNING: expected 3 (or 4 w/ alpha) channels")

    compare_to_examples(args.data_dir)
    print("\nRESULT:", "all splits fully paired" if ok else "SOME IMAGES MISSING (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
