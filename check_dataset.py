import pandas as pd
import numpy as np

network = "123Bus"

directory = "data/" + network

P = pd.read_csv(f"{directory}/S_real.csv").values
Q = pd.read_csv(f"{directory}/S_imag.csv").values
V_real = pd.read_csv(f"{directory}/V_real.csv").values
V_imag = pd.read_csv(f"{directory}/V_imag.csv").values
Y_real = pd.read_csv(f"{directory}/Y_real.csv", header=None).values
Y_imag = pd.read_csv(f"{directory}/Y_imag.csv", header=None).values

S = P + 1j * Q
V = V_real + 1j * V_imag
Ybus = Y_real + 1j * Y_imag

for i in range(V.shape[0]):
    I_calc = Ybus @ V[i]
    S_calc = V[i] * np.conj(I_calc)
    err = np.mean(np.abs(S[i] - S_calc))

    # breakpoint()
    assert err < 1e-1, f"Mismatch at sample {i}: error={err.item():.8f}"
