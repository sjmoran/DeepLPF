# -*- coding: utf-8 -*-
#Copyright (C) 2026. Sean Moran. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""Every parameter must receive a gradient from the training loss.

A parameter that no gradient reaches is trained by nobody: it keeps its
initialisation for the life of the run while appearing in the checkpoint and in
the parameter count. One forward pass and one backward is enough to find them,
and it is far cheaper than discovering it after a week on a GPU.
"""

import pytest
import torch

import model


def _starved_after_one_backward(net):
    """Names of parameters with no gradient after one backward pass."""
    net.train()
    prediction = net(torch.rand(1, 3, 48, 48))
    target = torch.rand_like(prediction)
    loss = model.DeepLPFLoss()(torch.clamp(prediction, 0, 1), target)
    if getattr(net, 'learn_filter_count', False):
        loss = loss + net.gate_penalty
    loss.backward()

    return {name for name, parameter in net.named_parameters()
            if parameter.grad is None or torch.count_nonzero(parameter.grad) == 0}


def test_no_parameter_is_starved():
    """The plain network must train every parameter it declares."""
    torch.manual_seed(0)

    starved = _starved_after_one_backward(model.DeepLPFNet())

    assert starved == set(), 'parameters receiving no gradient: %s' % sorted(starved)


@pytest.mark.parametrize('kwargs', [
    {'learn_filter_count': True},
    {'colour_knots': 16},
    {'colour_knots': 0},
    {'learn_filter_count': True, 'colour_knots': 16},
])
def test_no_parameter_is_starved_with_the_optional_heads(kwargs):
    """Every optional capability must also train everything it adds."""
    torch.manual_seed(0)

    starved = _starved_after_one_backward(model.DeepLPFNet(**kwargs))

    assert starved == set(), 'parameters receiving no gradient: %s' % sorted(starved)


def test_the_inversion_indicators_receive_a_gradient():
    """The graduated filter's inversion indicators must be trainable.

    They are binarised, and both the binarisation and the branch selection have
    to pass a gradient for the fully-connected layer that predicts them to
    learn anything.
    """
    torch.manual_seed(0)
    net = model.DeepLPFNet()

    starved = _starved_after_one_backward(net)

    assert 'deeplpfnet.graduated_filter.fc_graduated.weight' not in starved


def _graduated_row_gradients(seed):
    """Per-output-row gradient magnitude of ``fc_graduated.weight``, (24,)."""
    torch.manual_seed(seed)
    net = model.DeepLPFNet()
    net.train()
    prediction = net(torch.rand(1, 3, 48, 48))
    target = torch.rand_like(prediction)
    model.DeepLPFLoss()(torch.clamp(prediction, 0, 1), target).backward()
    weight = dict(net.named_parameters())[
        'deeplpfnet.graduated_filter.fc_graduated.weight']
    return weight.grad.abs().sum(dim=1)


SEEDS = range(8)


def test_no_graduated_output_is_structurally_starved():
    """No output of ``fc_graduated`` may be cut off from the loss by construction.

    There are two ways a row of this layer can show a zero gradient. One is
    ordinary saturation: the scale factors ([15:24]) pass through a clamp, and
    a value already outside its bounds passes nothing back on that step. That
    is the clamp doing its job, and it clears on the next batch.

    The other is structural. ``G[:, 6:9]`` used to be consumed only as
    ``y_axis_dist.data`` - a detached lower bound on the next line - so it set
    the forward value while receiving no gradient on any step, in any seed,
    ever. In the shipped checkpoint those three units are still at their
    initialisation, tanh01(0) = 0.5, and that untrained constant beat the
    trained offset on two of the three filter instances for every bundled
    example. Removing ``.data`` alone does not fix it: ``maximum`` routes the
    gradient to whichever argument it selected, so the pair take turns and
    each is still starved whenever the other wins.

    So the geometry outputs - the inversion indicator, the slope, the intercept
    and the two offsets, ``G[:, 0:15]`` - must carry gradient on *every* seed,
    and no output at all may be starved on every seed.
    """
    gradients = torch.stack([_graduated_row_gradients(s) for s in SEEDS])

    always_starved = [row for row in range(gradients.shape[1])
                      if not gradients[:, row].any()]
    assert always_starved == [], (
        'fc_graduated outputs receiving no gradient in any of %d seeds - a '
        'detached or unused path, not clamp saturation: %s'
        % (len(SEEDS), always_starved))

    geometry = gradients[:, 0:15]
    starved_somewhere = (geometry == 0).nonzero().tolist()
    assert starved_somewhere == [], (
        'line-geometry outputs G[:, 0:15] must receive gradient on every '
        'step; (seed index, output) pairs that did not: %s' % starved_somewhere)
