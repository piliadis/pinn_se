import pandas as pd
import numpy as np

directory = "data/13bus"
s_base = 1_000_000

# pu
# P_pu = -pd.read_csv(f"{directory}/real_S.csv").values / s_base
# P_pu[0:3] *= -1
# Q_pu = -pd.read_csv(f"{directory}/imag_S.csv").values / s_base
# Q_pu[0:3] *= -1
# V_mag_pu = pd.read_csv(f"{directory}/mag_V_pu.csv").values
# V_ang_rad = pd.read_csv(f"{directory}/ang_V_rad.csv").values
# Y_real_pu = pd.read_csv(f"{directory}/real_Y_pu.csv", header=None).values
# Y_imag_pu = pd.read_csv(f"{directory}/imag_Y_pu.csv", header=None).values

# S_pu = P_pu + 1j * Q_pu
# V_true_pu = V_mag_pu * np.exp(1j * V_ang_rad)
# Ybus_pu = Y_real_pu + 1j * Y_imag_pu

# for i in range(V_true_pu.shape[0]):
#     I_calc = Ybus_pu @ V_true_pu[i].T
#     S_calc = V_true_pu[i] * np.conj(I_calc)
#     err = np.mean(np.abs(S_pu[i] - S_calc))
#     breakpoint()
#     assert err < 1e-6, f"Mismatch at sample {i}: error={err.item():.8f}"

# SI

P = -pd.read_csv(f"{directory}/real_S.csv").values
P[0:3] *= -1
Q = -pd.read_csv(f"{directory}/imag_S.csv").values
Q[0:3] *= -1
V_mag = pd.read_csv(f"{directory}/mag_V.csv").values
V_ang_rad = pd.read_csv(f"{directory}/ang_V_rad.csv").values
Y_real = pd.read_csv(f"{directory}/real_Y_SI.csv", header=None).values
Y_imag = pd.read_csv(f"{directory}/imag_Y_SI.csv", header=None).values

S = P + 1j * Q
V_true = V_mag + 1j * V_ang_rad
Ybus = Y_real + 1j * Y_imag

for i in range(V_true.shape[0]):
    I_calc = Ybus @ V_true[i].T
    S_calc = V_true[i] * np.conj(I_calc)
    err = np.mean(np.abs(S[i] - S_calc))

    breakpoint()
    assert err < 1e-6, f"Mismatch at sample {i}: error={err.item():.8f}"


# Your original loss function
def pinn_loss(V_pred, V_true, I_true, Ybus, lambda_1=1.0, lambda_2=1.0):
    I_pred = np.matmul(Ybus, V_pred.T).T

    voltage_mse = np.mean(np.abs(V_pred - V_true) ** 2)
    voltage_mse_norm = voltage_mse / np.max(np.abs(V_pred - V_true) ** 2)

    current_mae = np.mean(np.abs(I_pred - I_true))
    current_mae_norm = current_mae / np.max(np.abs(I_pred - I_true))

    total_loss = lambda_1 * voltage_mse_norm + lambda_2 * current_mae_norm

    return total_loss, voltage_mse_norm, current_mae_norm


# Small function to verify physical correctness (I=Y*V)
def check_physical_correctness(V, I, Ybus):
    I_calc = np.matmul(Ybus, V.T).T
    error_max = np.max(np.abs(I - I_calc))
    error_min = np.min(np.abs(I - I_calc))
    error = np.mean(np.abs(I - I_calc))
    return error_max, error_min
