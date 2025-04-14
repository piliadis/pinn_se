import opendssdirect as dss
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

dss_file = "13Bus/IEEE13Nodeckt_clean.dss"
dss.Command("Clear")
dss.Command(f"compile [{dss_file}]")

dss.Text.Command("vsource.source.enabled=no")
dss.Text.Command("batchedit load..* enabled=no")
dss.Text.Command("batchedit capacitor..* enabled=no")
# dss.Text.Command("batchedit generator..* enabled=no")

dss.Command("solve")
assert dss.Solution.Converged(), "Solution did not converge!"

buses_order = dss.Circuit.YNodeOrder()  # SWSTO

# Extract Ybus matrix
data, indices, indptr = dss.YMatrix.getYsparse(factor=False)
Ybus = csc_matrix((data, indices, indptr)).toarray()

dss.Text.Command("Clear")
dss.Text.Command(f"compile [{dss_file}]")
dss.Text.Command("solve")
assert dss.Solution.Converged(), "Solution did not converge!"

buses = dss.Circuit.YNodeOrder()

# Extract voltages
V_array = dss.Circuit.YNodeVArray()
V_complex = np.array(
    [V_array[i] + 1j * V_array[i + 1] for i in range(0, len(V_array), 2)]
)  # ayto einai to swsto!!
V_df = pd.DataFrame([list(V_complex)], columns=buses)
V = V_df[buses_order].values.flatten()

# Extract currents NOTE: den eimai sigouros gia ayto
# I_array = dss.Circuit.YCurrents()
# I_array = np.array(
#     [I_array[i] + 1j * I_array[i + 1] for i in range(0, len(I_array), 2)]
# )
# I_df = pd.DataFrame([list(I_array)], columns=buses)
# I = I_df[buses_order].values.flatten()

I_calc = Ybus @ V
S_calc = V * np.conj(I_calc)
