import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model import FlexiblePINN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_data(directory):
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


def normalize_complex(data):
    """Normalize complex data to [0, 1] range by magnitude"""
    mag = torch.abs(data)
    min_mag = torch.min(mag)
    max_mag = torch.max(mag)
    
    # Normalize magnitude between 0 and 1
    mag_normalized = (mag - min_mag) / (max_mag - min_mag)
    
    # Keep angle unchanged
    angle = torch.angle(data)
    
    return mag_normalized * torch.exp(1j * angle), min_mag, max_mag


def denormalize_complex(data_normalized, min_mag, max_mag):
    """Denormalize complex data back to original scale"""
    mag_normalized = torch.abs(data_normalized)
    angle = torch.angle(data_normalized)
    
    # Denormalize magnitude
    mag = mag_normalized * (max_mag - min_mag) + min_mag
    
    return mag * torch.exp(1j * angle)


def pinn_loss(V_pred_norm, V_true_norm, Ybus, V_norm_params, lambda_1=1.0, lambda_2=1.0):
    """
    Calculate PINN loss with normalized inputs and proper scaling
    """
    # Voltage MSE Loss - calculate on normalized data
    voltage_mse_norm = torch.mean(torch.abs(V_pred_norm - V_true_norm) ** 2)
    
    # For physics constraint, denormalize voltages
    V_pred = denormalize_complex(V_pred_norm, *V_norm_params)
    V_true = denormalize_complex(V_true_norm, *V_norm_params)
    
    # Current calculation - use denormalized data for physical correctness
    I_pred = torch.matmul(Ybus, V_pred.T).T
    I_true = torch.matmul(Ybus, V_true.T).T
    
    # Use MAE for current (more stable than MSE)
    current_mae_denorm = torch.mean(torch.abs(I_pred - I_true))
    
    # Normalize current loss to make it comparable with voltage MSE
    # Use the maximum magnitude as reference (more stable than mean)
    current_scale = torch.max(torch.abs(I_true))
    current_mae_normalized = current_mae_denorm / current_scale
    
    # Combined loss
    total_loss = lambda_1 * voltage_mse_norm + lambda_2 * current_mae_normalized
    
    return total_loss, voltage_mse_norm, current_mae_normalized


if __name__ == "__main__":
    torch.manual_seed(42)

    network = "13Bus"

    data_dir = "data/" + network
    results_dir = "results/" + network
    os.makedirs(results_dir, exist_ok=True)

    # Load raw data
    S, V_true, I_true, Ybus = load_data(data_dir)

    # Normalize data (except Y_bus)
    S_norm, S_min, S_max = normalize_complex(S)
    # breakpoint()
    V_true_norm, V_min, V_max = normalize_complex(V_true)
    # Save normalization parameters
    V_norm_params = (V_min, V_max)

    num_samples, num_buses = S.shape
    model = FlexiblePINN(num_buses).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5
    )

    epochs = 10000

    # Initial weights for loss terms
    lambda_1, lambda_2 = 1.0, 0.0
    adjustment_step = 0.01
    adjustment_epochs = 200

    epoch_log, total_loss_log, voltage_loss_log, current_loss_log = [], [], [], []

    for epoch in range(epochs):
        model.train()
        # Input normalized power, get normalized voltage prediction
        V_mag_pred_norm, V_ang_pred_norm = model(S_norm)
        V_pred_norm = V_mag_pred_norm * torch.exp(1j * V_ang_pred_norm)

        total_loss, volt_loss, curr_loss = pinn_loss(
            V_pred_norm, V_true_norm, Ybus, V_norm_params,
            lambda_1=lambda_1, lambda_2=lambda_2
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        # Adjust lambda weights every 200 epochs
        # if (epoch + 1) % adjustment_epochs == 0:
        #     lambda_1 = max(0.5, lambda_1 - adjustment_step)
        #     lambda_2 = min(0.5, lambda_2 + adjustment_step)
        if curr_loss < 1_000:
            lambda_2 = 0.001
            lambda_1 = 1 - lambda_2
        else:
            lambda_2 = 0.0
            lambda_1 = 1


        # Logging
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}: Total Loss={total_loss.detach().item():.8f}, "
                f"Voltage Loss (normalized)={volt_loss.item():.8f}, Current Loss (normalized)={curr_loss.item():.8f}, "
                f'λ1={lambda_1:.2f}, λ2={lambda_2:.2f}, LR={optimizer.param_groups[0]["lr"]:.8f}'
            )
            epoch_log.append(epoch)
            total_loss_log.append(total_loss.item())
            voltage_loss_log.append(volt_loss.item())
            current_loss_log.append(curr_loss.item())

    # Final evaluation
    V_mag_final_norm, V_ang_final_norm = model(S_norm)
    V_final_pred_norm = V_mag_final_norm * torch.exp(1j * V_ang_final_norm)
    
    # Calculate final loss
    total_loss, volt_loss, curr_loss = pinn_loss(
        V_final_pred_norm, V_true_norm, Ybus, V_norm_params,
        lambda_1=lambda_1, lambda_2=lambda_2
    )

    print(f"Final Total Loss: {total_loss.item():.8f}")
    print(f"Final Voltage Loss (normalized): {volt_loss.item():.8f}")
    print(f"Final Current Loss (normalized): {curr_loss.item():.8f}")

    # Save final model explicitly
    torch.save({
        'model_state_dict': model.state_dict(),
        'normalization_params': {
            'V_min': V_min, 'V_max': V_max,
            'S_min': S_min, 'S_max': S_max
        }
    }, f"{results_dir}/pinn_model{adjustment_step}.pth")
    print("Model and normalization parameters saved")

    # Save logs to CSV
    logs = pd.DataFrame(
        {
            "Epoch": epoch_log,
            "Total Loss": total_loss_log,
            "Voltage Loss": voltage_loss_log,
            "Current Loss": current_loss_log,
        }
    )
    logs.to_csv(f"{results_dir}/loss_logs{adjustment_step}.csv", index=False)

    # Plot the error curves
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_log[1:], total_loss_log[1:], label="Total Loss", linewidth=2)
    plt.plot(
        epoch_log[1:],
        voltage_loss_log[1:],
        label="Normalized Voltage MSE",
        linewidth=2,
    )
    plt.plot(
        epoch_log[1:],
        current_loss_log[1:],
        label="Normalized Current MAE",
        linewidth=2,
    )
    plt.xlabel("Epochs")
    plt.ylabel("Normalized Error")
    plt.title("Voltage and Current Errors During Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/error_curves{adjustment_step}.png")
    plt.show()

    # Convert final predictions back to original scale for comparison
    V_final_pred_denorm = denormalize_complex(V_final_pred_norm, *V_norm_params)
    print("Predicted voltage magnitude (denormalized):", V_final_pred_denorm.abs()[0])
    print("True voltage magnitude (original):", V_true.abs()[0])

    breakpoint()