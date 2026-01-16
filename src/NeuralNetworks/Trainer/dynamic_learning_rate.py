# NeuralNetworks- Multi-Layer Perceptrons avec encodage Fourier
# Copyright (C) 2026 Alexandre Brun
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from ..Dependances import np

def generate_learning_rate (
    Nb_iter,
    X0,
    mode = "smoother", 
    first = 0.4,
    second = 1,
    Xi = 5e-4,
    Xf = 1e-6):

    infl = int (first * Nb_iter)
    Plat = int (second * Nb_iter)

    def smoothstep (x0, xa, n, m):
        values = []
        if m == "smooth":
            for i in range (n):
                t = i / (n - 1)          # t dans [0, 1]
                s = t * t * (3 - 2 * t)  # smoothstep
                x = x0 + (xa - x0) * s
                values.append (x)
        elif m == "smoother":
            for i in range(n):
                t = i / (n - 1)          # t dans [0, 1]
                s = t * t * t * (t * (6 * t - 15) + 10)
                x = x0 + (xa - x0) * s
                values.append(x)
        else:
            raise ValueError("mode doit être 'smooth' ou 'smoother'")
        return values
    
    cuv1 = smoothstep (X0, Xi, infl, mode)
    cuv2 = smoothstep (Xi, Xf, Plat - infl, mode)
    cuv3 = [Xf for _ in range (Plat, Nb_iter)]

    return np.array (cuv1 + cuv2 + cuv3)

def update_lr (losses, lrs, epoch, lr):

    loss = losses[-1] + (losses[-1] - losses[0])/len(losses)

    n = 9
    # Points de contrôle (multiplicité finale = dérivée nulle)
    P = np.array([
        0.0,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        1.0, 1.0
    ])
    
    # Coefficients binomiaux (précomputés UNE FOIS)
    C = np.array([1, 9, 36, 84, 126, 126, 84, 36, 9, 1], dtype=float)

    x = np.clip(loss, 0.0, 1.0)
    t = np.sqrt(x)

    u = 1.0 - t

    # Bernstein vectorisé
    y = (
        C[0] * u**9 * P[0] +
        C[1] * u**8 * t * P[1] +
        C[2] * u**7 * t**2 * P[2] +
        C[3] * u**6 * t**3 * P[3] +
        C[4] * u**5 * t**4 * P[4] +
        C[5] * u**4 * t**5 * P[5] +
        C[6] * u**3 * t**6 * P[6] +
        C[7] * u**2 * t**7 * P[7] +
        C[8] * u * t**8 * P[8] +
        C[9] * t**9 * P[9]
    )
    return np.clip (max(0.001 * y, lrs [epoch]), 0.0, lr)