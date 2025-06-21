import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from model import FlexiblePINN
from train_validation_test import load_data
from train_validation_test import robust_normalize_complex, robust_denormalize_complex


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
    run = "run_20250603_121047"

    val_data_dir = f"data_eval/{network}"
    model_path = f"results/{network}_noise/{run}/best_model.pth"

    # === Load Validation Data ===
    S_val, V_val, I_val, Ybus = load_data(val_data_dir)
    S_val, V_val, I_val, Ybus = [x.to(device) for x in (S_val, V_val, I_val, Ybus)]

    # ADD NOISE to S
    noise_level = 0.05  # 1% noise (adjustable)
    S_val = add_noise(S_val, noise_level=noise_level)

    # dropout_rate = 0.1  # Drop 20% of measurements
    # S_val_masked = randomly_mask_S(S_val, dropout_rate=dropout_rate)

    # === Initialize Model ===
    num_samples, num_buses = S_val.shape
    model = FlexiblePINN(num_buses).to(device).to(torch.float64)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    normalization_params = torch.load(
        f"results/{network}_noise/{run}/normalization_params.pt"
    )
    S_norm_params = normalization_params["S_norm_params"]
    V_norm_params = normalization_params["V_norm_params"]

    # === Predict and Evaluate ===
    with torch.no_grad():

        S_val_norm, _ = robust_normalize_complex(S_val, norm_params=S_norm_params)
        V_val_norm, _ = robust_normalize_complex(V_val, norm_params=V_norm_params)

        V_re_pred_norm, V_im_pred_norm = model(S_val_norm)
        V_pred_norm = torch.complex(V_re_pred_norm, V_im_pred_norm)

        volt_loss_norm = torch.mean(torch.abs(V_pred_norm - S_val_norm) ** 2)

        V_pred_phys = robust_denormalize_complex(V_pred_norm, *V_norm_params)
        volt_mse = torch.mean(torch.abs(V_pred_phys - V_val))

        I_pred = torch.matmul(Ybus, V_pred_phys.T).T
        curr_mse = torch.mean(torch.abs(I_pred - I_val))

        volt_rmse = torch.sqrt(torch.mean((V_pred_phys - V_val).abs() ** 2))
        curr_rmse = torch.sqrt(torch.mean((I_pred - I_val).abs() ** 2))

        # save one point results
        # V_pred_phys_cpu = S_val.detach().cpu()

        # # Compute magnitude and angle
        # V_mag = torch.abs(V_pred_phys_cpu[42])
        # V_ang_rad = torch.angle(V_pred_phys_cpu[42])
        # V_ang_deg = V_ang_rad * 180 / torch.pi  # Optional: convert to degrees

        # # Save each to CSV
        # output_dir = "results/csv_outputs"
        # os.makedirs(output_dir, exist_ok=True)

        # # Convert to DataFrame (each row is a sample, each column a bus)
        # df_mag = pd.DataFrame(V_mag.numpy())
        # df_ang = pd.DataFrame(V_ang_deg.numpy())  # or V_ang_rad.numpy() for radians

        # df_mag.to_csv(os.path.join(output_dir, "S_val_magnitude.csv"), index=False)
        # df_ang.to_csv(os.path.join(output_dir, "S_val_angle_deg.csv"), index=False)

        breakpoint()

        # save_complex_tensor(V_pred_phys[42], "V_pred_test", ".")
        # save_complex_tensor(V_val[42], "V_val", ".")
        # save_complex_tensor(I_pred[42], "I_pred_test", ".")
        # save_complex_tensor(I_val[42], "I_val", ".")

    test_results = {
        "Voltage Loss norm": volt_loss_norm.item(),
        "Voltage MAE": volt_mse.item(),
        "Current MAE": curr_mse.item(),
        "Voltage RMSE": volt_rmse.item(),
        "Current RMSE": curr_rmse.item(),
    }

    print("\n=== FINAL TEST RESULTS ===")
    for key, value in test_results.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")

    # # === Print Results ===
    # print("\nEvaluation Metrics:")
    # # print(f"Total Loss: {total_loss:.8f}")
    # # print(f"Voltage Loss (MSE): {volt_loss:.8f}")
    # # print(f"Current Loss (MAE): {curr_loss:.8f}")
    # print(f"Voltage Magnitude MAE: {mag_mae:.8f}")
    # print(f"Voltage Angle MAE: {ang_mae:.8f}")
    # print(f"Voltage Magnitude RMSE: {mag_rmse:.8f}")
    # print(f"Voltage Angle RMSE: {ang_rmse:.8f}")

    # # === Save metrics to CSV ===
    # os.makedirs(evaluation_results_dir, exist_ok=True)
    # metrics = {
    #     "Metric": [
    #         # "Total Loss",
    #         # "Voltage Loss (MSE)",
    #         # "Current Loss (MAE)",
    #         "Voltage Magnitude MAE",
    #         "Voltage Angle MAE",
    #         "Voltage Magnitude RMSE",
    #         "Voltage Angle RMSE",
    #     ],
    #     "Value": [
    #         # total_loss.item(),
    #         # volt_loss.item(),
    #         # curr_loss.item(),
    #         mag_mae,
    #         ang_mae,
    #         mag_rmse,
    #         ang_rmse,
    #     ],
    # }
    # df_metrics = pd.DataFrame(metrics)
    # df_metrics.to_csv(
    #     os.path.join(evaluation_results_dir, "evaluation_metrics.csv"), index=False
    # )

    # print(f"\nSaved metrics to {evaluation_results_dir}/evaluation_metrics.csv")

    # # === Plot and Save Error Histograms ===
    # plot_error_histograms(V_pred, V_true_val, evaluation_results_dir)

    # breakpoint()
