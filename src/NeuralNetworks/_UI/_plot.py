# NeuralNetworks - Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .._Dependances import np, plt, FixedLocator

def init_plot (fig_size : int = 5, color : str = "#888888"):
    
    plt.rcParams['axes.facecolor']   = (0,0,0,0)
    plt.rcParams ['text.color']      = color
    plt.rcParams ['axes.labelcolor'] = color
    plt.rcParams ['xtick.color']     = color
    plt.rcParams ['ytick.color']     = color
    plt.rcParams ['axes.edgecolor']  = color
    plt.rcParams ['axes.titlecolor'] = color
    plt.rcParams ['grid.color']      = color

    fig, ax = plt.subplots ()
    fig.set_figheight      (fig_size)
    fig.set_figwidth       (fig_size)

    return fig, ax

def plot (ax,
    x_title : str = "",
    y_title : str = "",
    title   : str = ""):

    plt.title     (title)
    ax.set_xlabel (x_title)
    ax.set_ylabel (y_title)
    ax.legend     (loc = "upper right", framealpha = 0, edgecolor = 'none')

    ticks, minors = list (ax.get_xticks ()), []; ticks [0] = 1
    for i in range( len (ticks) - 1):
        minors.extend (np.linspace (ticks [i], ticks [i+1], 4, endpoint = False) [1:])
    
    ax.set_xticks              (ticks)
    ax.xaxis.set_minor_locator (FixedLocator (minors))
    
    ax.grid    (which = "major", axis = "y", linewidth = 1.0)
    ax.grid    (which = "minor", axis = "y", linestyle = ":", alpha = 0.5)
    plt.yscale ('log', nonpositive='mask')
    
    plt.tight_layout     ()
    plt.ion              ()
    plt.show             ()