import pandapower as pp
import pandas as pd
import numpy as np
from cigre_LV_EU_residential import create_cigre_network_lv

# net = create_cigre_network_lv()
net = pp.networks.panda_four_load_branch()
# net = pp.networks.case33bw()


# load se MW/MVA
for i in range(20000):

    lower_bound = 0.01
    upper_bound = 0.1

    net.load["p_mw"] = np.random.uniform(lower_bound, upper_bound, 4).tolist()
    net.load["q_mvar"] = np.random.uniform(lower_bound, upper_bound, 4).tolist()

    pp.runpp(net)

    Ybus = net._ppc["internal"]["Ybus"].todense()
    Sbus = net._ppc["internal"]["Sbus"]
    V = net._ppc["internal"]["Sbus"]

    res = net.res_bus  # vm_pu, va_degree p_mw q_mvar

    # pd.DataFrame(Ybus.real).to_csv("real_Y.csv", index=False, header=False)
    # pd.DataFrame(Ybus.imag).to_csv("imag_Y.csv", index=False, header=False)

    # res["p_mw"].to_frame().T.to_csv("real_S.csv", index=False, header=False, mode="a")
    # res["q_mvar"].to_frame().T.to_csv("imag_S.csv", index=False, header=False, mode="a")

    # res["vm_pu"].to_frame().T.to_csv("mag_V.csv", index=False, header=False, mode="a")

    # df_radians = np.deg2rad(res["va_degree"])
    # df_radians.to_frame().T.to_csv("ang_V.csv", index=False, header=False, mode="a")

    print(i)

    breakpoint()
