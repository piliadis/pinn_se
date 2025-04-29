import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model import FlexiblePINN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Data loading (explicitly normalized)
def load_data_with_currents(directory):
    P = pd.read_csv(f"{directory}/S_real.csv").values
    Q = pd.read_csv(f"{directory}/S_imag.csv").values
    V_real = pd.read_csv(f"{directory}/V_real.csv").values
    V_imag = pd.read_csv(f"{directory}/V_imag.csv").values
    I_real = pd.read_csv(f"{directory}/I_real.csv").values
    I_imag = pd.read_csv(f"{directory}/I_imag.csv").values
    Y_real = pd.read_csv(f"{directory}/Y_real.csv", header=None).values
    Y_imag = pd.read_csv(f"{directory}/Y_imag.csv", header=None).values

    S = torch.tensor(P + 1j * Q, dtype=torch.cfloat, device=device)
    Ybus = torch.tensor(Y_real + 1j * Y_imag, dtype=torch.cfloat, device=device)
    V_true = torch.tensor(V_real + 1j * V_imag, dtype=torch.cfloat, device=device)
    I_true = torch.tensor(I_real + 1j * I_imag, dtype=torch.cfloat, device=device)

    return S, V_true, I_true, Ybus


if __name__ == "__main__":
    torch.manual_seed(42)

    network = "13Bus"

    data_dir = "data/" + network
    results_dir = "results/" + network

    S, V_true, I_true, Ybus = load_data_with_currents(data_dir)

    num_samples, num_buses = S.shape
    model = FlexiblePINN(num_buses).to(device)

    model.load_state_dict(
        torch.load(f"{results_dir}/pinn_model0.01.pth", weights_only=True)
    )

    V_mag_pred, V_ang_pred = model(S)

    print(V_mag_pred[0])
    print(V_true.real[0])

    breakpoint()
