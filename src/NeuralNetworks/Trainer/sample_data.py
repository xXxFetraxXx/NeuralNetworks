# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ..Dependances import train_test_split

def sample_data (inputs, outputs, test_size):
    if test_size is None:
        return inputs, inputs, outputs, outputs
    else:
        return train_test_split (
            inputs,
            outputs,
            test_size = test_size,
            random_state = 42
        )