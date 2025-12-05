# NeuralNetworksBeta - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2025 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import pytest
from src.NeuralNetworks import MLP, image

@pytest.fixture
def inputs_outputs():
    img, inputs, outputs = url("https://unesco.org.uk/site/assets/files/6266/the_forth_bridge_2.jpeg", 1)
    return img, inputs, outputs

@pytest.fixture
def net():
    return MLP([2, 1, 3])

def test_mlp_init(net):
    try:
        net
    except Exception as e:
        pytest.fail(f"MLP.__init__ failed with error: {e}")

def test_mlp_train(net, inputs_outputs):
    _, inputs, outputs = inputs_outputs
    try:
        net.train(inputs, outputs, 1, 1024)
    except Exception as e:
        pytest.fail(f"MLP.train() failed with error: {e}")

def test_mlp_call(net, inputs_outputs):
    _, inputs, _ = inputs_outputs
    try:
        net(inputs)
    except Exception as e:
        pytest.fail(f"MLP.__call__ failed with error: {e}")

def test_mlp_params(net):
    try:
        net.params()
    except Exception as e:
        pytest.fail(f"MLP.params() failed with error: {e}")

def test_mlp_neurons(net):
    try:
        net.neurons()
    except Exception as e:
        pytest.fail(f"MLP.neurons() failed with error: {e}")

def test_mlp_nb_params(net):
    try:
        net.nb_params()
    except Exception as e:
        pytest.fail(f"MLP.nb_params() failed with error: {e}")

def test_mlp_comparison_and_training(net, inputs_outputs):
    img, inputs, outputs = inputs_outputs
    net2 = MLP([2, 1, 3])

    try:
        image.plot(img, inputs, net, net2)
    except Exception as e:
        pytest.fail(f"plot(*nets) failed with error: {e}")

    try:
        image.compare(img, inputs, net, net2)
    except Exception as e:
        pytest.fail(f"compare(*nets) failed with error: {e}")

    try:
        losses(net, net2)
    except Exception as e:
        pytest.fail(f"losses(*nets) failed with error: {e}")

def test_mlp_invalid_init():
    try:
        net3 = MLP(optim="Random", crit="MSE", norm="Relu")
    except Exception as e:
        pytest.fail(f"MLP.__init__(optim='Random') failed with error: {e}")

    try:
        net3 = MLP(optim="Adam", crit="Random", norm="Relu")
    except Exception as e:
        pytest.fail(f"MLP.__init__(crit='Random') failed with error: {e}")

    try:
        net3 = MLP(optim="Adam", crit="MSE", norm="Random")
    except Exception as e:
        pytest.fail(f"MLP.__init__(norm='Random') failed with error: {e}")