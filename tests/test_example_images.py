# -*- coding: utf-8 -*-
#Copyright (C) 2026. Sean Moran. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""The bundled example inputs are inputs, not copies of their targets.

``adobe5k_dpe/deeplpf_example_test_input/a4774-_DGW0330.png`` once shipped as a
byte-for-byte copy of its own Expert C target (the same file is still wrong in
the published CURL release, so the mix-up predates this repo). Anything scored
against it is meaningless: the "input" already is the answer. The bundled
examples have since been regenerated from this repo's own export, but the test
stays: it catches the whole class - any example whose input and target are the
same picture - rather than that one id.
"""
import glob
import os

import numpy as np
import pytest
from PIL import Image

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'adobe5k_dpe')


def _by_id(dirname):
    """Map image id (the filename up to the first "-") to filepath."""
    paths = glob.glob(os.path.join(EX, dirname, '*.png'))
    return {os.path.basename(p).split('-')[0]: p for p in paths}


def _rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.int16)


INPUTS = _by_id('deeplpf_example_test_input')
TARGETS = _by_id('deeplpf_example_test_output')
PAIRED = sorted(set(INPUTS) & set(TARGETS))


@pytest.mark.parametrize('img_id', PAIRED)
def test_example_input_differs_from_its_target(img_id):
    inp, tgt = _rgb(INPUTS[img_id]), _rgb(TARGETS[img_id])
    assert inp.shape == tgt.shape, (
        '%s: input %s and target %s differ in size; both come from the same '
        'export and must match' % (img_id, inp.shape, tgt.shape))
    # A retouch always moves some pixels; identical files mean one was copied
    # over the other. The real pairs sit at a mean absolute difference of 8-25.
    assert np.abs(inp - tgt).mean() > 1.0, (
        '%s: example input is identical to its target' % img_id)


def test_examples_were_found():
    """Guard the guard: a renamed directory must not silently skip everything."""
    assert len(PAIRED) >= 5
