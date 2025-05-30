import pandas as pd
import numpy as np

network = "13Bus"

directory = "data/" + network

P = pd.read_csv(f"{directory}/S_real.csv").values
Q = pd.read_csv(f"{directory}/S_imag.csv").values
V_real = pd.read_csv(f"{directory}/V_real.csv").values
V_imag = pd.read_csv(f"{directory}/V_imag.csv").values
Y_real = pd.read_csv(f"{directory}/Y_real.csv", header=None).values
Y_imag = pd.read_csv(f"{directory}/Y_imag.csv", header=None).values
I_real = pd.read_csv(f"{directory}/I_real.csv").values
I_imag = pd.read_csv(f"{directory}/I_imag.csv").values

S = P + 1j * Q
V = V_real + 1j * V_imag
I = I_real + 1j * I_imag
Ybus = Y_real + 1j * Y_imag

# for i in range(V.shape[0]):
#     I_calc = Ybus @ V[i]
#     S_calc = V[i] * np.conj(I_calc)
#     err = np.mean(np.abs(S[i] - S_calc))

#     # breakpoint()
#     assert err < 1e-1, f"Mismatch at sample {i}: error={err.item():.8f}"


# Vectorized calculation
I_calc = V @ Ybus  # shape: [N, buses]
err1 = np.mean(np.abs(I - I_calc), axis=1)  # shape: [N]

assert np.all(err1 < 1e-1), f"Mismatches found! Max error = {np.max(err1):.8f}"


S_calc = V * np.conj(I_calc)  # element-wise complex power, shape: [N, buses]
err = np.mean(np.abs(S - S_calc), axis=1)  # shape: [N]

# Assert all errors are below threshold
assert np.all(err < 1e-1), f"Mismatches found! Max error = {np.max(err):.8f}"

print("All samples passed the power consistency check.")


breakpoint()
