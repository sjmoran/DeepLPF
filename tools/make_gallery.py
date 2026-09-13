#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Copyright (C) 2026. Sean Moran. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.
"""Regenerate ``images/gallery.jpg``, the README hero figure.

Four FiveK test photographs, one per row: the input, the filters the model
chose drawn over the input, the Expert C retouch, and the DeepLPF output with
its PSNR against Expert C.

The filter geometry is read straight out of the two parameter-prediction
layers with a forward hook, then decoded exactly as ``graduated.py`` and
``elliptical.py`` decode it. Both branches work in the normalised coordinates
of ``filtercommon._coord_grids``, where ``x_axis`` runs down the rows and
``y_axis`` across the columns -- so the "x" parameter of an ellipse is its
vertical centre, not its horizontal one.

Usage::

    python3 tools/make_gallery.py [--ids a4844 a4575 a4738 a4514]
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model  # noqa: E402
from util import ImageProcessing  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'adobe5k_dpe_data')
CKPT = os.path.join(REPO, 'pretrained_models', 'adobe_dpe',
                    'deeplpf_psnr_24.180_ssim_0.917_model.pt')

# The four photographs of the shipped figure, and the caption each row carries.
DEFAULT_IDS = ['a4844', 'a4575', 'a4738', 'a4514']
CAPTIONS = {
    'a4844': 'haze pulled off a city skyline',
    'a4575': 'a flat harbour given its blue back',
    'a4738': 'a swan lifted out of dark water',
    'a4514': 'an overcast waterfront opened up',
}

HEADERS = ['Input', 'The filters it chose', 'Expert C retouch', 'DeepLPF']
CYAN = '#22cff2'
PINK = '#ee5da8'

# Pixel geometry of the shipped figure (1800 x 1415), measured off it.
FIG_W, FIG_H, DPI = 1800, 1415, 100
COL_X = [(31, 467), (473, 908), (913, 1349), (1354, 1790)]
ROW_Y = [(39, 329), (377, 668), (717, 1007), (1056, 1346)]


def tanh01(x):
    return 0.5 * (np.tanh(x) + 1.0)


def find_image(kind, img_id):
    """Path of ``adobe5k_dpe_data/<kind>/<img_id>-*.png``."""
    d = os.path.join(DATA, kind)
    for name in sorted(os.listdir(d)):
        if name.split('-')[0] == img_id:
            return os.path.join(d, name)
    raise SystemExit('no %s image for id %s' % (kind, img_id))


def load(path):
    """Image as a (3, H, W) float tensor in [0, 1], as the data loader gives it."""
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def run(net, inp):
    """Enhanced image plus the raw 24 parameters of each filter branch."""
    grabbed = {}

    def hook(key):
        return lambda mod, args, out: grabbed.__setitem__(key, out.detach()[0].numpy())

    handles = [
        net.deeplpfnet.graduated_filter.fc_graduated.register_forward_hook(hook('grad')),
        net.deeplpfnet.elliptical_filter.fc_elliptical.register_forward_hook(hook('ell')),
    ]
    try:
        with torch.no_grad():
            out = torch.clamp(net(torch.clamp(inp.unsqueeze(0), 0, 1)), 0, 1)
    finally:
        for h in handles:
            h.remove()
    return out[0, 0:3].numpy(), grabbed['grad'], grabbed['ell']


def graduated_lines(G, W, H):
    """Central line of each graduated instance, as (xs, ys) in pixels.

    ``graduated.py`` places the line at ``y_axis = m * x_axis + c``, with
    ``y_axis`` the normalised column and ``x_axis`` the normalised row, so the
    line is swept over rows.
    """
    eps = 1e-10
    slope = G[3:6]
    c_lo = tanh01(G[6:9]) + eps
    c = np.clip(np.maximum(tanh01(G[9:12]), c_lo), None, 1.0)
    v = np.linspace(-2.0, 3.0, 2)
    return [((slope[i] * v + c[i]) * W, v * H) for i in range(3)]


def elliptical_curves(G, W, H):
    """Boundary of each elliptical instance, as (xs, ys) in pixels.

    ``elliptical.py`` measures the polar angle from the ``y_axis`` (column)
    direction and offsets it by the rotation ``A``; the radius at that angle
    is ``a b / sqrt(a^2 sin^2 + b^2 cos^2)``. The two normalised axes are
    scaled by W and H separately, so a circle in those coordinates comes out
    as an ellipse in pixels -- which is what the shipped figure shows.
    """
    eps = 1e-10
    row_c = tanh01(G[0:3]) + eps    # centre along x_axis, i.e. vertical
    col_c = tanh01(G[3:6]) + eps    # centre along y_axis, i.e. horizontal
    a = tanh01(G[6:9]) + eps
    b = tanh01(G[9:12]) + eps
    A = tanh01(G[12:15]) * math.pi + eps

    psi = np.linspace(0, 2 * math.pi, 361)
    out = []
    for i in range(3):
        phi = psi - A[i]
        r = (a[i] * b[i]) / np.sqrt((a[i] * np.sin(phi)) ** 2
                                    + (b[i] * np.cos(phi)) ** 2 + eps)
        out.append(((col_c[i] + r * np.cos(psi)) * W,
                    (row_c[i] + r * np.sin(psi)) * H))
    return out


def panel(fig, col, row, img_hw3):
    """Add an axes at the measured cell and show an (H, W, 3) array in it."""
    x0, x1 = COL_X[col]
    y0, y1 = ROW_Y[row]
    ax = fig.add_axes([x0 / FIG_W, 1 - y1 / FIG_H,
                       (x1 - x0) / FIG_W, (y1 - y0) / FIG_H])
    H, W = img_hw3.shape[:2]
    ax.imshow(img_hw3, extent=(0, W, H, 0), interpolation='lanczos')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ids', nargs=4, default=DEFAULT_IDS,
                    help='four FiveK ids, top row first')
    ap.add_argument('--checkpoint', default=CKPT)
    ap.add_argument('--out', default=os.path.join(REPO, 'images', 'gallery.jpg'))
    args = ap.parse_args()

    net = model.DeepLPFNet()
    net.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    net.eval()

    fig = plt.figure(figsize=(FIG_W / DPI, FIG_H / DPI), dpi=DPI,
                     facecolor='white')

    for col, header in enumerate(HEADERS):
        x0, x1 = COL_X[col]
        fig.text((x0 + x1) / 2 / FIG_W, 1 - 27 / FIG_H, header,
                 ha='center', va='center', fontsize=15, color='#222222')

    psnrs = []
    for row, img_id in enumerate(args.ids):
        inp = load(find_image('input', img_id))
        tgt = load(find_image('output', img_id))
        enh, G_grad, G_ell = run(net, inp)

        psnr = ImageProcessing.compute_psnr(
            tgt.numpy()[None].astype(np.float32),
            enh[None].astype(np.float32), 1.0)
        psnrs.append(psnr)

        inp_hw3 = inp.permute(1, 2, 0).numpy()
        H, W = inp_hw3.shape[:2]

        panel(fig, 0, row, inp_hw3)

        ax = panel(fig, 1, row, inp_hw3)
        for xs, ys in elliptical_curves(G_ell, W, H):
            ax.plot(xs, ys, color=PINK, lw=1.6, solid_capstyle='round')
        for xs, ys in graduated_lines(G_grad, W, H):
            ax.plot(xs, ys, color=CYAN, lw=1.6, solid_capstyle='round')

        panel(fig, 2, row, tgt.permute(1, 2, 0).numpy())

        ax = panel(fig, 3, row, np.transpose(enh, (1, 2, 0)))
        ax.text(0.975, 0.04, '%.1f dB' % psnr, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=14, color='white')

        y0, y1 = ROW_Y[row]
        fig.text(20 / FIG_W, 1 - (y0 + y1) / 2 / FIG_H, CAPTIONS.get(img_id, img_id),
                 rotation=90, ha='center', va='center', fontsize=13, color='#8a8c8e')

    # Legend: a short swatch and a label, twice, centred under the grid.
    ly = 1 - 1390 / FIG_H
    for x_line, x_text, colour, label in [
            (725, 770, CYAN, 'graduated filter'),
            (925, 968, PINK, 'elliptical filter')]:
        fig.add_artist(plt.Line2D([x_line / FIG_W, (x_line + 35) / FIG_W], [ly, ly],
                                  color=colour, lw=1.8))
        fig.text(x_text / FIG_W, ly, label, ha='left', va='center',
                 fontsize=13, color='#222222')

    fig.savefig(args.out, dpi=DPI, facecolor='white',
                pil_kwargs={'quality': 92, 'optimize': True})
    plt.close(fig)

    for img_id, psnr in zip(args.ids, psnrs):
        print('%s %.3f dB' % (img_id, psnr))
    print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
