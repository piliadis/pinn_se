"""
Vgazw tin generator kai vazw fortio sto 3
"""

import numpy as np
import pandas as pd
import os

# Directory to save results
output_dir = "data/3Bus"
os.makedirs(output_dir, exist_ok=True)

# Fixed admittances
y12, y13, y23 = 5 - 15j, 10 - 40j, 15 - 50j
Y = np.array(
    [[y12 + y13, -y12, -y13], [-y12, y12 + y23, -y23], [-y13, -y23, y13 + y23]],
    dtype=complex,
)
G = Y.real
B = Y.imag

iterations = 10000
results = {
    "V_real": [],
    "V_imag": [],
    "I_real": [],
    "I_imag": [],
    "S_real": [],
    "S_imag": [],
}
headers = ["Bus1", "Bus2", "Bus3"]

for sim in range(iterations):
    # Randomly vary load at Bus 2 and generation at Bus 3
    P2 = np.random.uniform(-1, -4)
    Q2 = np.random.uniform(-1, -0.1)
    P3 = np.random.uniform(-1, -4)
    Q3 = np.random.uniform(-1, -0.1)
    # P2 = -2.0
    # Q2 = -0.5
    # P3 = -1.5
    # Q3 = -0.7

    P_spec = np.array([0.0, P2, P3])
    Q_spec = np.array([0.0, Q2, Q3])

    V = np.array([1.0, 1.0, 1.0])
    delta = np.zeros(3)

    pv, pq, pv_pq = [], [1, 2], [1, 2]
    max_iter, tol = 10, 1e-6

    for iteration in range(max_iter):
        # V_complex = V * np.exp(1j * delta)
        # I = Ybus @ V_complex
        # S = V_complex * np.conj(I)
        P_calc = np.zeros(3)
        Q_calc = np.zeros(3)
        for i in range(3):
            for k in range(3):
                angle_diff = delta[i] - delta[k]
                P_calc[i] += (
                    V[i]
                    * V[k]
                    * (G[i, k] * np.cos(angle_diff) + B[i, k] * np.sin(angle_diff))
                )
                Q_calc[i] += (
                    V[i]
                    * V[k]
                    * (G[i, k] * np.sin(angle_diff) - B[i, k] * np.cos(angle_diff))
                )

        # Compute mismatches
        dP = P_spec[pv_pq] - P_calc[pv_pq]
        dQ = Q_spec[pq] - Q_calc[pq]
        mismatch = np.concatenate((dP, dQ))

        if np.max(np.abs(mismatch)) < tol:
            break

        # Jacobian matrix elements
        J11 = np.zeros((len(pv_pq), len(pv_pq)))
        J12 = np.zeros((len(pv_pq), len(pq)))
        J21 = np.zeros((len(pq), len(pv_pq)))
        J22 = np.zeros((len(pq), len(pq)))

        for i, ii in enumerate(pv_pq):
            for j, jj in enumerate(pv_pq):
                if ii == jj:
                    sum_term = 0
                    for k in range(3):
                        if k != ii:
                            angle = delta[ii] - delta[k]
                            sum_term += V[k] * (
                                G[ii, k] * np.sin(angle) - B[ii, k] * np.cos(angle)
                            )
                    J11[i, j] = -Q_calc[ii] - V[ii] ** 2 * B[ii, ii]
                else:
                    angle = delta[ii] - delta[jj]
                    J11[i, j] = (
                        V[ii]
                        * V[jj]
                        * (G[ii, jj] * np.sin(angle) - B[ii, jj] * np.cos(angle))
                    )

        for i, ii in enumerate(pv_pq):
            for j, jj in enumerate(pq):
                if ii == jj:
                    sum_term = 0
                    for k in range(3):
                        angle = delta[ii] - delta[k]
                        sum_term += V[k] * (
                            G[ii, k] * np.cos(angle) + B[ii, k] * np.sin(angle)
                        )
                    J12[i, j] = sum_term + V[ii] * G[ii, ii]
                else:
                    angle = delta[ii] - delta[jj]
                    J12[i, j] = V[ii] * (
                        G[ii, jj] * np.cos(angle) + B[ii, jj] * np.sin(angle)
                    )

        for i, ii in enumerate(pq):
            for j, jj in enumerate(pv_pq):
                if ii == jj:
                    sum_term = 0
                    for k in range(3):
                        angle = delta[ii] - delta[k]
                        sum_term += V[k] * (
                            G[ii, k] * np.cos(angle) + B[ii, k] * np.sin(angle)
                        )
                    J21[i, j] = P_calc[ii] - V[ii] ** 2 * G[ii, ii]
                else:
                    angle = delta[ii] - delta[jj]
                    J21[i, j] = (
                        -V[ii]
                        * V[jj]
                        * (G[ii, jj] * np.cos(angle) + B[ii, jj] * np.sin(angle))
                    )

        for i, ii in enumerate(pq):
            for j, jj in enumerate(pq):
                if ii == jj:
                    sum_term = 0
                    for k in range(3):
                        angle = delta[ii] - delta[k]
                        sum_term += V[k] * (
                            G[ii, k] * np.sin(angle) - B[ii, k] * np.cos(angle)
                        )
                    J22[i, j] = -Q_calc[ii] - V[ii] * B[ii, ii]
                else:
                    angle = delta[ii] - delta[jj]
                    J22[i, j] = V[ii] * (
                        G[ii, jj] * np.sin(angle) - B[ii, jj] * np.cos(angle)
                    )

        # Assemble Jacobian matrix
        J_top = np.hstack((J11, J12))
        J_bottom = np.hstack((J21, J22))
        J = np.vstack((J_top, J_bottom))

        # Solve for updates
        dx = np.linalg.solve(J, mismatch)

        # Update variables
        delta[pv_pq] += dx[0 : len(pv_pq)]
        V[pq] += dx[len(pv_pq) :]

    V_complex = V * np.exp(1j * delta)
    I = Y @ V_complex
    S = V_complex * np.conj(I)

    results["V_real"].append(V_complex.real)
    results["V_imag"].append(V_complex.imag)
    results["I_real"].append(I.real)
    results["I_imag"].append(I.imag)
    results["S_real"].append(S.real)
    results["S_imag"].append(S.imag)


# Utility function to write CSV
def write_csv(filename, headers, data):
    pd.DataFrame(data, columns=headers).to_csv(filename, index=False)


np.savetxt(f"{output_dir}/Y_real.csv", Y.real, delimiter=",")
np.savetxt(f"{output_dir}/Y_imag.csv", Y.imag, delimiter=",")
write_csv(f"{output_dir}/V_real.csv", headers, results["V_real"])
write_csv(f"{output_dir}/V_imag.csv", headers, results["V_imag"])
write_csv(f"{output_dir}/I_real.csv", headers, results["I_real"])
write_csv(f"{output_dir}/I_imag.csv", headers, results["I_imag"])
write_csv(f"{output_dir}/S_real.csv", headers, results["S_real"])
write_csv(f"{output_dir}/S_imag.csv", headers, results["S_imag"])

print("Simulations completed. Results saved to CSV files.")
