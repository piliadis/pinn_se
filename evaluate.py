import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from model import FlexiblePINN
from train import load_data, zscore_normalize_complex, zscore_denormalize_complex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def compute_metrics(V_pred, V_true):
    V_true_mag = torch.abs(V_true)
    V_true_ang = torch.angle(V_true)
    V_pred_mag = torch.abs(V_pred)
    V_pred_ang = torch.angle(V_pred)

    mag_mae = torch.mean(torch.abs(V_pred_mag - V_true_mag)).item()
    ang_mae = torch.mean(torch.abs(V_pred_ang - V_true_ang)).item()

    mag_rmse = torch.sqrt(torch.mean((V_pred_mag - V_true_mag) ** 2)).item()
    ang_rmse = torch.sqrt(torch.mean((V_pred_ang - V_true_ang) ** 2)).item()

    return mag_mae, ang_mae, mag_rmse, ang_rmse


def plot_error_histograms(V_pred, V_true, results_dir):
    V_true_mag = torch.abs(V_true).cpu().numpy()
    V_true_ang = torch.angle(V_true).cpu().numpy()
    V_pred_mag = torch.abs(V_pred).cpu().numpy()
    V_pred_ang = torch.angle(V_pred).cpu().numpy()

    mag_errors = np.abs(V_pred_mag - V_true_mag).flatten()
    ang_errors = np.abs(V_pred_ang - V_true_ang).flatten()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(mag_errors, bins=50, edgecolor="black")
    plt.xlabel("Voltage Magnitude Error (pu)")
    plt.ylabel("Frequency")
    plt.title("Voltage Magnitude Error Distribution")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(ang_errors, bins=50, edgecolor="black")
    plt.xlabel("Voltage Angle Error (rad)")
    plt.ylabel("Frequency")
    plt.title("Voltage Angle Error Distribution")
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "error_histograms.png"))
    plt.show()


def add_noise(tensor, noise_level=0.01):
    # Uniform noise
    noise_real = (
        torch.empty_like(tensor.real).uniform_(-noise_level, noise_level) * tensor.real
    )
    noise_imag = (
        torch.empty_like(tensor.imag).uniform_(-noise_level, noise_level) * tensor.imag
    )
    # Gausian noise
    # noise_real = torch.randn_like(tensor.real) * noise_level * tensor.real
    # noise_imag = torch.randn_like(tensor.imag) * noise_level * tensor.imag
    noisy_tensor = (tensor.real + noise_real) + 1j * (tensor.imag + noise_imag)
    return noisy_tensor


def randomly_mask_S(S, dropout_rate=0.2):
    """
    Randomly zero out a fraction of the elements in S (complex tensor).
    dropout_rate: fraction of elements to set to 0 (e.g., 0.2 = 20%)
    """
    mask = torch.rand_like(S.real) > dropout_rate
    S_masked = (S.real * mask) + 1j * (S.imag * mask)
    return S_masked


if __name__ == "__main__":
    network = "13Bus"
    run = "run_20240530_092500"

    val_data_dir = f"data/{network}"
    model_path = f"results/{network}/{run}/model.pth"
    evaluation_results_dir = f"results/{network}/evaluation"

    # === Load Validation Data ===
    S_val, V_true_val, I_true_val, Ybus_val = load_data(val_data_dir)

    # === OPTIONAL: Keep only first X samples ===
    X = 1  # You can change to 100, 500, etc.
    S_val = S_val[:X]
    S_val1 = torch.randn_like(S_val)
    V_true_val = V_true_val[:X]
    I_true_val = I_true_val[:X]

    # ADD NOISE to S
    noise_level = 0.50  # 1% noise (adjustable)
    S_val_noisy = add_noise(S_val, noise_level=noise_level)

    # dropout_rate = 0.2  # Drop 20% of measurements
    # S_val_masked = randomly_mask_S(S_val, dropout_rate=dropout_rate)

    breakpoint()

    # === Initialize Model ===
    num_samples, num_buses = S_val.shape
    model = FlexiblePINN(num_buses).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    normalization_params = torch.load(
        f"results/{network}/{run}/normalization_params.pt"
    )

    V_true_norm, *V_norm_params = zscore_normalize_complex(V_true_val, *V_norm_params)

    # === Predict and Evaluate ===
    with torch.no_grad():
        V_re_pred_norm, V_ang_pred = model(S_val1)
        V_pred = V_mag_pred + 1j * V_ang_pred

        breakpoint()

        # total_loss, volt_loss, curr_loss = pinn_loss(
        #     V_pred, V_true_val, I_true_val, Ybus_val, lambda_1=1.0, lambda_2=0.0
        # )

        mag_mae, ang_mae, mag_rmse, ang_rmse = compute_metrics(V_pred, V_true_val)

    # === Print Results ===
    print("\nEvaluation Metrics:")
    # print(f"Total Loss: {total_loss:.8f}")
    # print(f"Voltage Loss (MSE): {volt_loss:.8f}")
    # print(f"Current Loss (MAE): {curr_loss:.8f}")
    print(f"Voltage Magnitude MAE: {mag_mae:.8f}")
    print(f"Voltage Angle MAE: {ang_mae:.8f}")
    print(f"Voltage Magnitude RMSE: {mag_rmse:.8f}")
    print(f"Voltage Angle RMSE: {ang_rmse:.8f}")

    # === Save metrics to CSV ===
    os.makedirs(evaluation_results_dir, exist_ok=True)
    metrics = {
        "Metric": [
            # "Total Loss",
            # "Voltage Loss (MSE)",
            # "Current Loss (MAE)",
            "Voltage Magnitude MAE",
            "Voltage Angle MAE",
            "Voltage Magnitude RMSE",
            "Voltage Angle RMSE",
        ],
        "Value": [
            # total_loss.item(),
            # volt_loss.item(),
            # curr_loss.item(),
            mag_mae,
            ang_mae,
            mag_rmse,
            ang_rmse,
        ],
    }
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(
        os.path.join(evaluation_results_dir, "evaluation_metrics.csv"), index=False
    )

    print(f"\nSaved metrics to {evaluation_results_dir}/evaluation_metrics.csv")

    # === Plot and Save Error Histograms ===
    plot_error_histograms(V_pred, V_true_val, evaluation_results_dir)

    breakpoint()
