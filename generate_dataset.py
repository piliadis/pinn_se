import opendssdirect as dss
import numpy as np
import pandas as pd
import csv
import os
import random
from scipy.sparse import csc_matrix

network = "mynetwork"

# se kathe fakelo exw ena myscript.dss pou kleinw tous tap changers.
dss_file = "opendss/" + network + "/myscript.dss"

num_runs = 10000
variation_low, variation_high = 0.8, 1.2
initial_dir = os.getcwd()
output_dir = "data/" + network

# ~~~~~~~~~~~~~~~~~~~~ GET Y ~~~~~~~~~~~~~~~~~~~~~
dss.Command("Clear")
dss.Command(f"compile [{dss_file}]")

dss.Text.Command("vsource.source.enabled=no")
dss.Text.Command("batchedit load..* enabled=no")
dss.Text.Command("batchedit capacitor..* enabled=no")
# dss.Text.Command("batchedit generator..* enabled=no")

dss.Command("solve")
assert dss.Solution.Converged(), "Solution did not converge!"

y_node_order = dss.Circuit.YNodeOrder()  # SWSTO

# Extract Ybus matrix
data, indices, indptr = dss.YMatrix.getYsparse(factor=False)
Ybus = csc_matrix((data, indices, indptr)).toarray()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

headers = y_node_order

# Containers for results
results = {
    key: [] for key in ["V_real", "V_imag", "I_real", "I_imag", "S_real", "S_imag"]
}

# Simulation Loop
for run in range(num_runs):
    dss.Text.Command("Clear")
    dss.Text.Command(f"compile [{initial_dir+'/'+dss_file}]")
    # breakpoint()

    # === Randomize Load ===
    for load in dss.Loads.AllNames():
        dss.Loads.Name(load)
        factor = random.uniform(variation_low, variation_high)
        dss.Loads.kW(dss.Loads.kW() * factor)
        dss.Loads.kvar(dss.Loads.kvar() * factor)

    # === Solve Power Flow ===
    dss.Command("solve")
    if not dss.Solution.Converged():
        print(f"Run {run + 1} did not converge")
        continue

    buses = dss.Circuit.YNodeOrder()

    # === Extract Voltages ===
    V_array = dss.Circuit.YNodeVArray()
    V_complex = np.array(
        [V_array[i] + 1j * V_array[i + 1] for i in range(0, len(V_array), 2)]
    )  # ayto einai to swsto!!
    V_df = pd.DataFrame([list(V_complex)], columns=buses)
    V = V_df[y_node_order].values.flatten()

    # === Power Calculation ===
    I_calc = Ybus @ V
    S_calc = V * np.conj(I_calc)

    # === Store Results ===
    results["V_real"].append(V.real)
    results["V_imag"].append(V.imag)
    results["I_real"].append(I_calc.real)
    results["I_imag"].append(I_calc.imag)
    results["S_real"].append(S_calc.real)
    results["S_imag"].append(S_calc.imag)

    # === Progress Update ===
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
np.savetxt(f"{output_dir}/Y_real.csv", Ybus.real, delimiter=",")
np.savetxt(f"{output_dir}/Y_imag.csv", Ybus.imag, delimiter=",")
write_csv(f"{output_dir}/V_real.csv", headers, results["V_real"])
write_csv(f"{output_dir}/V_imag.csv", headers, results["V_imag"])
write_csv(f"{output_dir}/I_real.csv", headers, results["I_real"])
write_csv(f"{output_dir}/I_imag.csv", headers, results["I_imag"])
write_csv(f"{output_dir}/S_real.csv", headers, results["S_real"])
write_csv(f"{output_dir}/S_imag.csv", headers, results["S_imag"])


print("Simulation complete. CSV files generated.")
