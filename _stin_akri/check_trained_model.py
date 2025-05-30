import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model import FlexiblePINN
from train_with_correct_norm import (
    normalize_complex_position_wise,
    load_data,
    denormalize_complex_position_wise,
)

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

    S, V_true, I_true, Ybus = load_data(data_dir)

    num_samples, num_buses = S.shape
    model = FlexiblePINN(num_buses).to(device)

    checkpoint = torch.load(f"{results_dir}/pinn_model1e-05.pth")

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set model to evaluation mode

    # Retrieve normalization parameters
    normalization_params = checkpoint["normalization_params"]
    V_min_mag = normalization_params["V_min_mag"]
    V_max_mag = normalization_params["V_max_mag"]
    S_min_mag = normalization_params["S_min_mag"]
    S_max_mag = normalization_params["S_max_mag"]

    print("Model and normalization parameters loaded")

    # Normalize data position-wise
    # Normalize data position-wise
    S_norm, S_min_mag, S_max_mag = normalize_complex_position_wise(S)
    V_true_norm, V_min_mag, V_max_mag = normalize_complex_position_wise(V_true)

    # Save normalization parameters
    V_norm_params = (V_min_mag, V_max_mag)
    S_norm_params = (S_min_mag, S_max_mag)

    V_mag_pred_norm, V_ang_pred_norm = model(S_norm)
    V_pred_norm = V_mag_pred_norm * torch.exp(1j * V_ang_pred_norm)

    V_final_pred_denorm = denormalize_complex_position_wise(V_pred_norm, *V_norm_params)

    breakpoint()
    # print(V_mag_pred[0])
    # print(V_true.real[0])
