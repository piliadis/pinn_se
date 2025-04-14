import opendssdirect as dss
import numpy as np
import csv
import os
import random
from scipy.sparse import csc_matrix

# Configuration
num_runs = 10000
variation_low, variation_high = 0.8, 1.2
dss_file = "13Bus/IEEE13Nodeckt.dss"
initial_dir = os.getcwd()
output_dir = "data/13bus"
os.makedirs(output_dir, exist_ok=True)

S_BASE = 1_000_000  # [VA]

# Compile Circuit
dss.Command("Clear")
dss.Command(f"compile [{dss_file}]")

# Get Node Order and Base Voltages
y_node_order = dss.Circuit.YNodeOrder()
headers = y_node_order

# Get Base Loads
base_loads = {
    load: (dss.Loads.kW(), dss.Loads.kvar())
    for load in dss.Loads.AllNames()
    if dss.Loads.Name(load)
}

# Containers for results
results = {key: [] for key in ["ang_V_rad", "mag_V", "mag_V_pu", "real_S", "imag_S"]}

# Get V_base per node
V_base = []  # [V]
for bus_phase in y_node_order:
    dss.Circuit.SetActiveBus(bus_phase)
    v_base = dss.Bus.kVBase() * 1000  # convert from kV to V
    V_base.append(v_base)

# Simulation Loop
for run in range(num_runs):

    # === Randomize Load ===
    for load, (kw, kvar) in base_loads.items():
        dss.Loads.Name(load)
        factor = random.uniform(variation_low, variation_high)
        dss.Loads.kW(kw * factor)
        dss.Loads.kvar(kvar * factor)

    # === Solve Power Flow ===
    dss.Command("solve")
    if not dss.Solution.Converged():
        print(f"Run {run + 1} did not converge")
        continue

    # === Extract Voltages ===
    voltages = dss.Circuit.AllBusVolts()
    voltages_complex = np.array(
        [voltages[i] + 1j * voltages[i + 1] for i in range(0, len(voltages), 2)]
    )
    mag_V = np.abs(voltages_complex)
    ang_V_rad = np.angle(voltages_complex)
    mag_V_pu = mag_V / np.array(V_base)

    # === Power Calculation ===
    power_dict = {
        bus.lower(): np.zeros(3, dtype=complex) for bus in dss.Circuit.AllBusNames()
    }

    for elem in dss.Circuit.AllElementNames():
        dss.Circuit.SetActiveElement(elem)
        powers = dss.CktElement.Powers()
        buses = [b.split(".")[0].lower() for b in dss.CktElement.BusNames()]
        nph, nterm = dss.CktElement.NumPhases(), dss.CktElement.NumTerminals()

        for t, bus in enumerate(buses):
            sign = 1 if t == 0 else -1
            for ph in range(nph):
                idx = 2 * (t * nph + ph)
                if idx + 1 < len(powers):
                    power_dict[bus][ph] += sign * (powers[idx] + 1j * powers[idx + 1])

    # Source power adjustment
    dss.Vsources.First()
    source_bus = dss.CktElement.BusNames()[0].split(".")[0].lower()
    source_power = dss.CktElement.Powers()
    for ph in range(3):
        idx = 2 * ph
        if idx + 1 < len(source_power):
            power_dict[source_bus][ph] -= source_power[idx] + 1j * source_power[idx + 1]

    # Reorder power based on YNodeOrder
    real_S, imag_S = [], []
    for node in y_node_order:
        bus, ph = node.split(".")
        ph_idx = int(ph) - 1
        S = power_dict.get(bus.lower(), [0, 0, 0])[ph_idx] * 1000  # [W, VAR]
        real_S.append(S.real)
        imag_S.append(S.imag)

    # === Store Results ===
    results["ang_V_rad"].append(ang_V_rad)
    results["mag_V"].append(mag_V)
    results["mag_V_pu"].append(mag_V_pu)
    results["real_S"].append(real_S)
    results["imag_S"].append(imag_S)

    # === Calculate and Save Ybus (first run only) ===
    if run == 0:
        dss.Circuit.SystemY()
        data, indices, indptr = dss.YMatrix.getYsparse(factor=False)
        n = len(y_node_order)
        Y_sparse = csc_matrix((data, indices, indptr), shape=(n, n)).toarray()
        np.savetxt("real_Y_SI.csv", Y_sparse.real, delimiter=",")
        np.savetxt("imag_Y_SI.csv", Y_sparse.imag, delimiter=",")

        # Y in per unit
        Z_base = (np.array(V_base) ** 2) / S_BASE  # assuming 1 MVA base
        Y_pu = Y_sparse * Z_base[:, None]
        np.savetxt("real_Y_pu.csv", Y_pu.real, delimiter=",")
        np.savetxt("imag_Y_pu.csv", Y_pu.imag, delimiter=",")
        np.savetxt("V_base.csv", V_base, delimiter=",")

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
write_csv(f"{output_dir}/ang_V_rad.csv", headers, results["ang_V_rad"])
write_csv(f"{output_dir}/mag_V.csv", headers, results["mag_V"])
write_csv(f"{output_dir}/mag_V_pu.csv", headers, results["mag_V_pu"])
write_csv(f"{output_dir}/real_S.csv", headers, results["real_S"])
write_csv(f"{output_dir}/imag_S.csv", headers, results["imag_S"])


print("Simulation complete. CSV files generated.")
