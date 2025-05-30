"""
23/05/2025
- vgainoun ola ektos apo tous 2 tranformers
- epeidi den vgike akri me tous trafos, eftiaksa ena modified ieee13 me ena voltage level.
vgainei ok to pu. kai ekana kanonika ta runs sta opoia eida oti to current loss vgainei pali
poly megalo (~30m) enw einai pu.. To pio a
"""

import opendssdirect as dss
import numpy as np
import pandas as pd
import csv
import os
import random
from scipy.sparse import csc_matrix

network = "13Bus"

dss_file = "opendss/" + network + "/IEEE13Nodeckt_simple.dss"

num_runs = 2000
variation_low, variation_high = 0.8, 1.2
initial_dir = os.getcwd()
output_dir = "data/" + network

S_base = 100e3  # 100 kVA

# ~~~~~~~~~~~~~~~~~~~~ GET Y ~~~~~~~~~~~~~~~~~~~~~
dss.Basic.ClearAll()
dss.Text.Command(f"compile [{dss_file}]")

dss.Text.Command("vsource.source.enabled=no")
dss.Text.Command("batchedit load..* enabled=no")

dss.Solution.Solve()
assert dss.Solution.Converged(), "Initial solution did not converge."

# Extract Ybus matrix
data, indices, indptr = dss.YMatrix.getYsparse(factor=False)
Ybus_SI = csc_matrix((data, indices, indptr)).toarray()

# Extract base voltages for each node
V_base_dict = {}
for bus in dss.Circuit.AllBusNames():
    dss.Circuit.SetActiveBus(bus)  # make this bus the active one
    V_base_dict[bus.upper()] = (
        dss.Bus.kVBase() * 1e3  # * np.sqrt(3)
    )  # Convert kV to V -> LL !!


def convert_Y_to_pu(Y_SI, V_base_dict, node_order):
    """
    Convert the admittance matrix Y_SI to per-unit using node-specific voltage bases.

    Parameters:
    -----------
    Y_SI : np.ndarray
        Admittance matrix in Siemens (shape: [n, n], complex dtype).
    V_base_dict : dict
        Dictionary mapping node names to base voltage in volts, e.g., {"650": 7200.0, ...}.
    node_order : list
        List of node names in the same order as used in Y_SI.

    Returns:
    --------
    Y_pu : np.ndarray
        Admittance matrix in per unit.
    """
    n = len(node_order)
    V_base_array = np.array(
        [V_base_dict[node.split(".")[0].upper()] for node in node_order]
    )

    V_base_matrix = np.zeros_like(Y_SI, dtype=complex)
    for i in range(n):
        for j in range(n):
            if V_base_array[i] == V_base_array[j]:
                V_base_matrix[i, j] = V_base_array[i]
            else:
                # breakpoint()
                V_base_matrix[i, j] = max(V_base_array[i], V_base_array[j])

    # Prepare the header: an empty string for the top-left cell, followed by node labels
    header = "," + ",".join(node_order)

    # Combine node labels as the first column with the data
    # Convert node_order to a column vector
    node_order_column = np.array(node_order).reshape(-1, 1)

    # Combine the node labels with the real part of the Y_base_matrix
    data_with_labels = np.hstack((node_order_column, V_base_matrix.real.astype(str)))

    # np.savetxt(
    #     "V_base.csv",
    #     data_with_labels,
    #     fmt="%s",
    #     delimiter=",",
    #     header=header,
    #     comments="",
    # )
    # breakpoint()
    Z_base_matrix = (V_base_matrix**2) / S_base  # in Ohms
    Y_base_matrix = 1 / Z_base_matrix  # in Siemens
    Ybus_pu = Y_SI / Y_base_matrix

    # Stack real and imaginary parts side by side
    # real_part = Ybus_pu.real
    # imag_part = Ybus_pu.imag
    # combined = np.empty((Ybus_pu.shape[0], Ybus_pu.shape[1] * 2), dtype=real_part.dtype)
    # combined[:, 0::2] = real_part
    # combined[:, 1::2] = imag_part
    # np.savetxt("Ybus_pu.csv", combined, delimiter=",", fmt="%.6e")

    # data = np.loadtxt("Ybus_pu.csv", delimiter=",")
    # real_parts = data[:, 0::2]  # Even-indexed columns
    # imag_parts = data[:, 1::2]  # Odd-indexed columns
    # Ybus_pu = real_parts + 1j * imag_parts
    return Ybus_pu


y_node_order = dss.Circuit.YNodeOrder()

Ybus_pu = convert_Y_to_pu(Ybus_SI, V_base_dict, y_node_order)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Prepare headers
headers = y_node_order

# Containers for results
results = {
    key: []
    for key in [
        "V_real_pu",
        "V_imag_pu",
        "I_real_pu",
        "I_imag_pu",
        "S_real_pu",
        "S_imag_pu",
    ]
}

# Simulation Loop
for run in range(num_runs):
    dss.Basic.ClearAll()
    dss.Text.Command(f"compile [{initial_dir+'/'+dss_file}]")

    # Randomize Load
    for load in dss.Loads.AllNames():
        dss.Loads.Name(load)
        factor = random.uniform(variation_low, variation_high)
        dss.Loads.kW(dss.Loads.kW() * factor)
        dss.Loads.kvar(dss.Loads.kvar() * factor)

    # Solve Power Flow
    dss.Solution.Solve()
    if not dss.Solution.Converged():
        print(f"Run {run + 1} did not converge")
        continue

    buses = dss.Circuit.YNodeOrder()

    # Extract Voltages
    V_array = dss.Circuit.YNodeVArray()  # exei allaksei to order kai thelei ftiaksimo!!
    V_complex = np.array(
        [V_array[i] + 1j * V_array[i + 1] for i in range(0, len(V_array), 2)]
    )  # ayto einai to swsto!!
    V_df = pd.DataFrame([list(V_complex)], columns=buses)
    V = V_df[y_node_order].values.flatten()
    V_pu = V / np.array(
        [V_base_dict[bus.split(".")[0].upper()] for bus in y_node_order]
    )  # Convert to per unit

    # Compute Currents and Powers
    I_calc = Ybus_SI @ V
    S_calc = V * np.conj(I_calc)
    I_pu = Ybus_pu @ V_pu
    S_pu = V_pu * np.conj(I_pu)

    # # Verification: Compare with SI
    I_calc2 = I_pu * np.array(
        [S_base / (V_base_dict[bus.split(".")[0].upper()]) for bus in y_node_order]
    )
    df = pd.DataFrame(
        {
            "y_node_order": y_node_order,
            "I_calc.real": I_calc.real,
            "I_calc.imag": I_calc.imag,
            "I_calc2.real": I_calc2.real,
            "I_calc2.imag": I_calc2.imag,
            "real_diff": np.abs(I_calc2.real - I_calc.real),
            "real_equal": np.abs(I_calc2.real - I_calc.real) < 0.1,
        }
    )
    # df.to_csv("I_.csv", index=False)
    # os.startfile("I_.csv")

    # if not np.allclose(I_calc2, I_calc, atol=1e-3):
    #     print(f"Discrepancy in currents at run {run + 1}")
    #     breakpoint()
    #     continue

    S_calc2 = S_pu * S_base
    df = pd.DataFrame(
        {
            "y_node_order": y_node_order,
            "S_calc.real": S_calc.real,
            "S_calc.imag": S_calc.imag,
            "S_calc2.real": S_calc2.real,
            "S_calc2.imag": S_calc2.imag,
            "real_diff": np.abs(S_calc2.real - S_calc.real),
            "real_equal": np.abs(S_calc2.real - S_calc.real) < 0.1,
        }
    )
    # df.to_csv("S_.csv", index=False)
    # os.startfile("S_.csv")
    # if not np.allclose(S_calc2, S_calc, atol=1e-3):
    #     print(f"Discrepancy in powers at run {run + 1}")
    #     breakpoint()
    #     continue

    # Store Results
    results["V_real_pu"].append(V_pu.real)
    results["V_imag_pu"].append(V_pu.imag)
    results["I_real_pu"].append(I_pu.real)
    results["I_imag_pu"].append(I_pu.imag)
    results["S_real_pu"].append(S_pu.real)
    results["S_imag_pu"].append(S_pu.imag)

    # Progress Update
    if (run + 1) % 1000 == 0:
        print(f"Completed {run + 1}/{num_runs}")


# === Export Results ===
def write_csv(filename, header, data):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)


os.chdir(initial_dir)
os.makedirs(output_dir, exist_ok=True)
np.savetxt(os.path.join(output_dir, "Y_real_pu.csv"), Ybus_pu.real, delimiter=",")
np.savetxt(os.path.join(output_dir, "Y_imag_pu.csv"), Ybus_pu.imag, delimiter=",")
write_csv(os.path.join(output_dir, "V_real_pu.csv"), headers, results["V_real_pu"])
write_csv(os.path.join(output_dir, "V_imag_pu.csv"), headers, results["V_imag_pu"])
write_csv(os.path.join(output_dir, "I_real_pu.csv"), headers, results["I_real_pu"])
write_csv(os.path.join(output_dir, "I_imag_pu.csv"), headers, results["I_imag_pu"])
write_csv(os.path.join(output_dir, "S_real_pu.csv"), headers, results["S_real_pu"])
write_csv(os.path.join(output_dir, "S_imag_pu.csv"), headers, results["S_imag_pu"])

print("Simulation complete. CSV files generated.")
