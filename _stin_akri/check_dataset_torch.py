"""
Oi prakseis me complex apaitoun float64!!!
"""

import pandas as pd
import torch

network = "13Bus"
directory = "data/" + network

# Load data from CSV files
P = torch.tensor(pd.read_csv(f"{directory}/S_real.csv").values, dtype=torch.float64)
Q = torch.tensor(pd.read_csv(f"{directory}/S_imag.csv").values, dtype=torch.float64)
V_real = torch.tensor(
    pd.read_csv(f"{directory}/V_real.csv").values, dtype=torch.float64
)
V_imag = torch.tensor(
    pd.read_csv(f"{directory}/V_imag.csv").values, dtype=torch.float64
)
Y_real = torch.tensor(
    pd.read_csv(f"{directory}/Y_real.csv", header=None).values, dtype=torch.float64
)
Y_imag = torch.tensor(
    pd.read_csv(f"{directory}/Y_imag.csv", header=None).values, dtype=torch.float64
)
I_real = torch.tensor(
    pd.read_csv(f"{directory}/I_real.csv").values, dtype=torch.float64
)
I_imag = torch.tensor(
    pd.read_csv(f"{directory}/I_imag.csv").values, dtype=torch.float64
)


# Combine into complex tensors
S = torch.complex(P, Q)  # shape: [N, buses]
V = torch.complex(V_real, V_imag)  # shape: [N, buses]
I = torch.complex(I_real, I_imag)  # shape: [N, buses]
Ybus = torch.complex(Y_real, Y_imag)  # shape: [buses, buses]


# for i in range(V.shape[0]):
#     I_calc = Ybus @ V[i]
#     S_calc = V[i] * torch.conj(I_calc)
#     err = torch.mean(torch.abs(S[i] - S_calc))

#     # breakpoint()
#     assert err < 1e-1, f"Mismatch at sample {i}: error={err.item():.8f}"


# Vectorized calculation
I_calc = torch.matmul(V, Ybus)  # shape: [N, buses]
# I_calc = torch.matmul(Ybus, V.T).T  # NOTE: to idio

err1 = torch.mean(torch.abs(I - I_calc), axis=1)  # shape: [N]

assert torch.all(err1 < 1e-1), f"Mismatches found! Max error = {torch.max(err1):.8f}"


S_calc = V * torch.conj(I_calc)  # element-wise complex power, shape: [N, buses]
err = torch.mean(torch.abs(S - S_calc), dim=1)  # shape: [N]

# Assert all errors are below threshold
assert torch.all(
    err < 1e-1
), f"Mismatches found! Max error = {torch.max(err).item():.8f}"

print("All samples passed the power consistency check.")


breakpoint()
