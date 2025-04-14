import opendssdirect as dss
import numpy as np
from scipy.sparse import csc_matrix

# Compile and solve the circuit
dss_file = "minimal.dss"
dss.Text.Command("Clear")
dss.Text.Command(f"compile [{dss_file}]")
dss.Text.Command("solve")

# Extract Ybus matrix
data, indices, indptr = dss.YMatrix.getYsparse(factor=False)
Ybus = csc_matrix((data, indices, indptr)).toarray()

# Extract voltages
V_array = dss.Circuit.YNodeVArray()
V_complex = np.array(
    [V_array[i] + 1j * V_array[i + 1] for i in range(0, len(V_array), 2)]
)  # ayto einai to swsto!!

# Extract currents NOTE: den eimai sigouros gia ayto
I_array = dss.Circuit.YCurrents()
I_array = np.array(
    [I_array[i] + 1j * I_array[i + 1] for i in range(0, len(I_array), 2)]
)


# Calculate currents and power
I_calc = Ybus @ V_complex
S_calc = V_complex * np.conj(I_calc)

# Output results
for i, node in enumerate(dss.Circuit.YNodeOrder()):
    print(f"Node: {node}")
    print(f"V (V): {V_complex[i]}")
    print(f"I (A): {I_calc[i]}")
    print(f"S_calc (VA): {S_calc[i]}")

breakpoint()
