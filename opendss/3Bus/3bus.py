"""
--------------------------------------------------------------
Newton-Raphson Power Flow Solver for a 3-Bus System (Example 3)
--------------------------------------------------------------

This script implements the Newton-Raphson method to solve the
power flow equations of a 3-bus system as described in Example 3
of the IntechOpen chapter titled:

  "Power Flow Analysis" - https://www.intechopen.com/chapters/65445

System Description:
-------------------
- Bus 1: Slack Bus
    Voltage magnitude: 1.02 pu
    Voltage angle: 0 degrees (reference)

- Bus 2: PQ Bus
    Active power demand: -2.0 pu
    Reactive power demand: -0.5 pu

- Bus 3: PV Bus
    Active power injection: 1.5 pu
    Voltage magnitude: 1.03 pu
    (Reactive power Q3 is to be determined)

Line admittances (in per unit):
    y12 = 5 - 15j
    y13 = 10 - 40j
    y23 = 15 - 50j

Methodology:
------------
- Construct the bus admittance matrix Ybus
- Initialize voltage magnitudes and angles
- At each iteration:
    - Compute power mismatches (ΔP, ΔQ)
    - Build Jacobian matrix
    - Solve for Δδ and Δ|V| using linear system
    - Update voltage angles and magnitudes
- Repeat until mismatches are below the specified tolerance

Assumptions:
------------
- Newton-Raphson is used in polar coordinates
- Two unknown angles (δ2, δ3), one unknown magnitude (|V2|)
- Tolerance and maximum number of iterations are defined

Author: ChatGPT (based on IntechOpen Example)
Date: 19/5/2025
--------------------------------------------------------------
"""

import numpy as np

# Bus types: 0 - Slack, 1 - PQ, 2 - PV
bus_type = [0, 1, 2]

# Specified power values (pu)
P_spec = np.array([0.0, -2.0, 1.5])  # P1 (Slack), P2 (PQ), P3 (PV)
Q_spec = np.array([0.0, -0.5, 0.0])  # Q1 (Slack), Q2 (PQ), Q3 (PV - to be calculated)

# Initial voltage magnitudes (pu) and angles (radians)
V = np.array([1.02, 1.0, 1.03])  # Initial guesses
delta = np.radians([0.0, 0.0, 0.0])  # Initial angles

# Line admittances (pu)
y12 = 5 - 15j
y13 = 10 - 40j
y23 = 15 - 50j

# Construct Y-bus matrix
Y = np.array(
    [[y12 + y13, -y12, -y13], [-y12, y12 + y23, -y23], [-y13, -y23, y13 + y23]],
    dtype=complex,
)

G = Y.real
B = Y.imag

# Indices for PV and PQ buses
pv = [2]  # Bus 3
pq = [1]  # Bus 2
pv_pq = pq + pv  # Buses 2 and 3

# Newton-Raphson iteration parameters
max_iter = 10
tolerance = 1e-8

for iteration in range(max_iter):
    # Calculate P and Q
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

    # Check for convergence
    if np.max(np.abs(mismatch)) < tolerance:
        print(f"Converged in {iteration} iterations.")
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

# Display results
print("\nFinal Bus Voltages and Angles:")
for i in range(3):
    print(f"Bus {i+1}: |V| = {V[i]:.5f} pu, δ = {np.degrees(delta[i]):.4f}°")

Vcomplex = V * np.exp(1j * delta)

breakpoint()
