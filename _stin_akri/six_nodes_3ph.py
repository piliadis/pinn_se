""" """

import pandas as pd
import numpy as np
from utils import *


S_base = 1_000
V_base = 400 / np.sqrt(3)
Y_base = S_base / V_base**2  # 6.25


directory = "data/mynetwork/"
P = pd.read_csv(f"{directory}/S_real.csv").values
Q = pd.read_csv(f"{directory}/S_imag.csv").values
V_real = pd.read_csv(f"{directory}/V_real.csv").values
V_imag = pd.read_csv(f"{directory}/V_imag.csv").values
# I_real = pd.read_csv(f"{directory}/I_real.csv").values
# I_imag = pd.read_csv(f"{directory}/I_imag.csv").values
Y_real = pd.read_csv(f"{directory}/Y_real.csv", header=None).values
Y_imag = pd.read_csv(f"{directory}/Y_imag.csv", header=None).values

# S = P + 1j * Q
# Ybus = Y_real + 1j * Y_imag
V_true = V_real + 1j * V_imag
# I_true = I_real + 1j * I_imag

Vmag_pu = np.abs(V_true).T / V_base
# breakpoint()

P_3ph = (P / S_base).T
Q_3ph = (Q / S_base).T
# P_3ph = (np.abs(P) / S_base).T
# Q_3ph = (np.abs(Q) / S_base).T

# Step 5: estimate admittance
delta_V = Vmag_pu - 1
G_hat = (P_3ph / Vmag_pu) @ delta_V.T @ np.linalg.pinv(delta_V @ delta_V.T)
B_hat = -(Q_3ph / Vmag_pu) @ delta_V.T @ np.linalg.pinv(delta_V @ delta_V.T)

G_hat *= Y_base
B_hat *= Y_base

N = P_3ph.shape[0]

# breakpoint()
# plot_heatmap(G_hat)
# plot_heatmap(B_hat)
plot_2_heatmaps(Y_real, G_hat)
plot_2_heatmaps(Y_imag, B_hat)

# === Clean G_hat and B_hat ===
# Force symmetry keeping zero entries
for i in range(N):
    for j in range(N):
        if G_hat[i, j] == 0:
            G_hat[j, i] = 0
        if B_hat[i, j] == 0:
            B_hat[j, i] = 0

# Clip invalid signs
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        if G_hat[i, j] > 0:
            G_hat[i, j] = 0
        if B_hat[i, j] < 0:
            B_hat[i, j] = 0

# Enforce symmetry by keeping the more conservative value
for i in range(N):
    for j in range(N):
        if G_hat[i, j] != G_hat[j, i]:
            m = min(G_hat[i, j], G_hat[j, i])
            G_hat[i, j] = G_hat[j, i] = m
        if B_hat[i, j] != B_hat[j, i]:
            m = max(B_hat[i, j], B_hat[j, i])
            B_hat[i, j] = B_hat[j, i] = m


# Zero-out small entries
# NOISE_THRESHOLD_G = -3
# NOISE_THRESHOLD_B = 10
# for i in range(N):
#     for j in range(N):
#         if NOISE_THRESHOLD_G < G_hat[i, j] < 0:
#             G_hat[i, j] = G_hat[j, i] = 0
#         if B_hat[i, j] < NOISE_THRESHOLD_B:
#             B_hat[i, j] = B_hat[j, i] = 0

# Fix diagonals
for i in range(N):
    G_hat[i, i] = -np.sum(G_hat[i, :]) + G_hat[i, i]
    B_hat[i, i] = -np.sum(B_hat[i, :]) + B_hat[i, i]

plot_2_heatmaps(Y_real, G_hat)
plot_2_heatmaps(Y_imag, B_hat)
# plot_heatmap(G_hat)
# plot_heatmap(B_hat)
