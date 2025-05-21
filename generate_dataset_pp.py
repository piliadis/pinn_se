import pandapower as pp
import pandas as pd
import numpy as np
import os
import csv


network = "33Bus"
num_runs = 10000
variation_low, variation_high = 0.8, 1.2
output_dir = f"data/{network}"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# net = pp.networks.panda_four_load_branch()
net = pp.networks.case33bw()

# Get bus names in correct order
bus_order = list(net.bus.name)

# Run once to get Ybus and node ordering
pp.runpp(net)
Ybus = net._ppc["internal"]["Ybus"].todense()
np.savetxt(f"{output_dir}/Y_real.csv", Ybus.real, delimiter=",")
np.savetxt(f"{output_dir}/Y_imag.csv", Ybus.imag, delimiter=",")


# Containers for results
results = {
    key: [] for key in ["V_real", "V_imag", "I_real", "I_imag", "S_real", "S_imag"]
}

p_nom = net.load["p_mw"]
q_nom = net.load["q_mvar"]


# load se MW/MVA
for run in range(num_runs):
    # Randomize loads
    net.load["p_mw"] = p_nom.apply(
        lambda x: x * np.random.uniform(variation_low, variation_high)
    )
    net.load["q_mvar"] = q_nom.apply(
        lambda x: x * np.random.uniform(variation_low, variation_high)
    )

    pp.runpp(net)

    res = net.res_bus

    # Get voltage per bus
    V_mag = res.vm_pu.values  # NOTE: in p.u.
    V_ang = np.deg2rad(res.va_degree.values)
    V_complex = V_mag * np.exp(1j * V_ang)  # shape (n,)

    # Compute I and S
    I_calc = Ybus @ V_complex  # shape (1, 6)
    I_calc = np.asarray(I_calc).flatten()  # ensure it's 1D even if it wasn’t

    S_calc = V_complex * np.conj(I_calc)

    results["V_real"].append(V_complex.real)
    results["V_imag"].append(V_complex.imag)
    results["I_real"].append(I_calc.real)
    results["I_imag"].append(I_calc.imag)
    results["S_real"].append(S_calc.real)
    results["S_imag"].append(S_calc.imag)

    if (run + 1) % 1000 == 0:
        print(f"Completed {run + 1}/{num_runs}")


# === Export CSVs ===
def write_csv(filename, header, data):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


write_csv(f"{output_dir}/V_real.csv", bus_order, results["V_real"])
write_csv(f"{output_dir}/V_imag.csv", bus_order, results["V_imag"])
write_csv(f"{output_dir}/I_real.csv", bus_order, results["I_real"])
write_csv(f"{output_dir}/I_imag.csv", bus_order, results["I_imag"])
write_csv(f"{output_dir}/S_real.csv", bus_order, results["S_real"])
write_csv(f"{output_dir}/S_imag.csv", bus_order, results["S_imag"])

print("Simulation complete. CSV files generated.")
