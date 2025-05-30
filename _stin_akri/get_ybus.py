"""
To eidame me ton Pano stis 23/5 kai einai numerically unstable.
"""

import numpy as np
from utils import load_data


network = "13Bus"

data_dir = "data/" + network

S, V_true, I_true, Ybus = load_data(data_dir)  # 10000 timestamps
# V_true (1000,41)
# I_true (1000,41)

breakpoint()

# prwta tsekaroume oti i methodos doulevei gia SI
Y_est = I_true.T @ np.linalg.pinv(V_true.T)
Y_est[np.abs(Y_est) < 1e-5] = 0
np.savetxt("Y_est.csv", Y_est.real, delimiter=",")

if not np.allclose(V_true, Y_est, atol=1e-3):
    breakpoint()


# meta ypologizoume to Y_pu
V_base_array = np.array(
    [
        115000.0,
        115000.0,
        115000.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        480.0,
        480.0,
        480.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
        4160.0,
    ]
)

I_base_array = np.array(
    [
        0.86956522,
        0.86956522,
        0.86956522,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        208.33333333,
        208.33333333,
        208.33333333,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
        24.03846154,
    ]
)

V_pu = V_true / V_base_array
I_pu = I_true / I_base_array


Y_pu_est = I_pu.T @ np.linalg.pinv(V_pu.T)
np.savetxt("Y_pu_est.csv", Y_pu_est.real, delimiter=",")

breakpoint()
