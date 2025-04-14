import opendssdirect as dss
import numpy as np
from scipy.sparse import csc_matrix

# === Minimal DSS model ===
minimal_dss = """
Clear
New Circuit.Minimal basekV=12.47 bus1=source phases=3
Set controlmode=off

New Transformer.T1 phases=3 windings=2 buses=[source, bus1]
~ conns=[delta, wye] kvs=[12.47, 0.48] kvas=[500, 500] %Rs=[0.5, 0.5] Xhl=1

New Linecode.MyLine nphases=3 r1=0.01 x1=0.085 r0=0.03 x0=0.15 c1=3.4 c0=1.6 units=mi

New Line.L1 bus1=bus1 bus2=load length=0.01 linecode=MyLine phases=3

New Load.Load1 bus1=load.1.2.3 phases=3 conn=wye kV=0.48 kW=30 kvar=15

Solve
"""

# === Compile circuit ===
dss.Text.Command(minimal_dss)
assert dss.Solution.Converged(), "Power flow did not converge."

# === Ybus and voltage extraction ===
dss.Circuit.SystemY()
y_node_order = dss.Circuit.YNodeOrder()
data, indices, indptr = dss.YMatrix.getYsparse(factor=False)
Ybus = csc_matrix(
    (data, indices, indptr), shape=(len(y_node_order), len(y_node_order))
).toarray()

# === Extract voltages matching YNodeOrder ===
V = []
for node in y_node_order:
    bus, ph = node.split(".")
    dss.Circuit.SetActiveBus(bus)
    voltages = dss.Bus.Voltages()
    nodes = dss.Bus.Nodes()
    try:
        idx = nodes.index(int(ph))
        v = voltages[2 * idx] + 1j * voltages[2 * idx + 1]
        V.append(v)
    except ValueError:
        V.append(0 + 0j)

V = np.array(V)
V_kV = V / 1000

# === Compute S = V * conj(YV) ===
I = Ybus @ V_kV
S_calc = V_kV * np.conj(I) * 1e6  # [VA]

# === Extract true power from elements ===
power_dict = {
    bus.lower(): np.zeros(3, dtype=complex) for bus in dss.Circuit.AllBusNames()
}
for elem in dss.Circuit.AllElementNames():
    dss.Circuit.SetActiveElement(elem)
    powers = dss.CktElement.Powers()
    buses = [b.split(".")[0].lower() for b in dss.CktElement.BusNames()]
    nph = dss.CktElement.NumPhases()
    for t, bus in enumerate(buses):
        sign = 1 if t == 0 else -1
        for ph in range(nph):
            idx = 2 * (t * nph + ph)
            if idx + 1 < len(powers):
                power_dict[bus][ph] += sign * (powers[idx] + 1j * powers[idx + 1])

# Adjust for source
dss.Vsources.First()
source_bus = dss.CktElement.BusNames()[0].split(".")[0].lower()
powers = dss.CktElement.Powers()
for ph in range(3):
    idx = 2 * ph
    if idx + 1 < len(powers):
        power_dict[source_bus][ph] -= powers[idx] + 1j * powers[idx + 1]

# Order S_true by y_node_order
S_true = []
for node in y_node_order:
    bus, ph = node.split(".")
    ph_idx = int(ph) - 1
    val = power_dict.get(bus.lower(), [0, 0, 0])[ph_idx] * 1000  # [VA]
    S_true.append(val)

S_true = np.array(S_true)

# === Compare ===
error = np.abs(S_calc - S_true)
print("Max error:", np.max(error))
print("Mean error:", np.mean(error))
assert np.allclose(S_calc, S_true, atol=1e-1), "Mismatch in power balance!"
