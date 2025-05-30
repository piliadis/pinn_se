import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

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


def normalize_complex_position_wise(data):
    """
    Normalize complex data position-wise across all samples.
    Each position in the vector is normalized separately using stats from all samples.

    Args:
        data: tensor of shape [num_samples, vector_length] with complex values

    Returns:
        normalized data, min_mag (per position), max_mag (per position)
    """
    # Extract magnitude and phase
    mag = torch.abs(data)
    angle = torch.angle(data)

    # Calculate min and max magnitude for each position across all samples
    min_mag = torch.min(mag, dim=0)[0]  # Shape: [vector_length]
    max_mag = torch.max(mag, dim=0)[0]  # Shape: [vector_length]

    # Avoid division by zero with small epsilon
    epsilon = 1e-8
    scale = max_mag - min_mag + epsilon

    # Normalize magnitude for each position
    mag_normalized = (mag - min_mag) / scale

    # Reconstruct complex numbers with normalized magnitude
    data_normalized = mag_normalized * torch.exp(1j * angle)

    return data_normalized, min_mag, max_mag


def denormalize_complex_position_wise(data_normalized, min_mag, max_mag):
    """
    Denormalize complex data that was normalized position-wise.

    Args:
        data_normalized: normalized complex tensor
        min_mag: min magnitude per position
        max_mag: max magnitude per position

    Returns:
        denormalized complex tensor
    """
    # Extract magnitude and phase
    mag_normalized = torch.abs(data_normalized)
    angle = torch.angle(data_normalized)

    # Calculate scale
    epsilon = 1e-8
    scale = max_mag - min_mag + epsilon

    # Denormalize magnitude
    mag = mag_normalized * scale + min_mag

    # Reconstruct complex numbers
    return mag * torch.exp(1j * angle)


def pinn_loss(
    V_pred_norm, V_true_norm, Ybus, V_norm_params, lambda_1=1.0, lambda_2=1.0
):
    """
    Calculate PINN loss with normalized inputs and proper scaling
    """
    # Voltage MSE Loss - calculate on normalized data
    voltage_mse_norm = torch.mean(torch.abs(V_pred_norm - V_true_norm) ** 2)

    # For physics constraint, denormalize voltages
    V_min_mag, V_max_mag = V_norm_params
    V_pred = denormalize_complex_position_wise(V_pred_norm, V_min_mag, V_max_mag)
    V_true = denormalize_complex_position_wise(V_true_norm, V_min_mag, V_max_mag)

    # Current calculation using physics
    I_pred = torch.matmul(Ybus, V_pred.T).T
    I_true = torch.matmul(Ybus, V_true.T).T

    # Current loss with normalization
    current_mae = torch.mean(torch.abs(I_pred - I_true))

    # Scale current loss to be comparable with voltage loss
    current_scale = torch.mean(torch.abs(I_true))
    current_loss_normalized = current_mae / (current_scale + 1e-8)

    # Combined loss
    total_loss = lambda_1 * voltage_mse_norm + lambda_2 * current_loss_normalized

    return total_loss, voltage_mse_norm, current_loss_normalized


class EarlyStopping:
    """Early stopping to terminate training when validation loss doesn't improve."""

    def __init__(self, patience=100, min_delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save model
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(
                    f"Early stopping triggered after {self.patience} epochs without improvement"
                )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    network = "13Bus"

    data_dir = "data/" + network
    results_dir = "results/" + network
    os.makedirs(results_dir, exist_ok=True)

    # Load raw data
    S, V_true, I_true, Ybus = load_data(data_dir)

    # Get the total number of samples
    num_samples, num_buses = S.shape

    # Create shuffled indices
    indices = np.random.permutation(num_samples)

    # Calculate split points
    train_end = int(0.8 * num_samples)
    val_end = int(0.9 * num_samples)

    # Split indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create data splits
    S_train = S[train_indices]
    V_true_train = V_true[train_indices]

    S_val = S[val_indices]
    V_true_val = V_true[val_indices]

    S_test = S[test_indices]
    V_true_test = V_true[test_indices]

    print(
        f"Data split: Train {len(train_indices)}, Validation {len(val_indices)}, Test {len(test_indices)}"
    )

    # Normalize data position-wise (using only training data statistics for fair evaluation)
    S_train_norm, S_min_mag, S_max_mag = normalize_complex_position_wise(S_train)
    V_true_train_norm, V_min_mag, V_max_mag = normalize_complex_position_wise(
        V_true_train
    )

    # Normalize validation and test sets using training set statistics
    mag_val = torch.abs(S_val)
    angle_val = torch.angle(S_val)
    S_val_norm = ((mag_val - S_min_mag) / (S_max_mag - S_min_mag + 1e-8)) * torch.exp(
        1j * angle_val
    )

    mag_test = torch.abs(S_test)
    angle_test = torch.angle(S_test)
    S_test_norm = ((mag_test - S_min_mag) / (S_max_mag - S_min_mag + 1e-8)) * torch.exp(
        1j * angle_test
    )

    mag_val = torch.abs(V_true_val)
    angle_val = torch.angle(V_true_val)
    V_true_val_norm = (
        (mag_val - V_min_mag) / (V_max_mag - V_min_mag + 1e-8)
    ) * torch.exp(1j * angle_val)

    mag_test = torch.abs(V_true_test)
    angle_test = torch.angle(V_true_test)
    V_true_test_norm = (
        (mag_test - V_min_mag) / (V_max_mag - V_min_mag + 1e-8)
    ) * torch.exp(1j * angle_test)

    # Save normalization parameters
    V_norm_params = (V_min_mag, V_max_mag)
    S_norm_params = (S_min_mag, S_max_mag)

    # Initialize model
    model = FlexiblePINN(num_buses).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=2000, min_delta=1e-6, path=f"{results_dir}/best_model.pt"
    )

    epochs = 10000

    # Initial weights for loss terms
    lambda_1, lambda_2 = 1.0, 0.0
    adjustment_step = 0.05
    adjustment_epochs = 500

    epoch_log = []
    train_loss_log = []
    val_loss_log = []
    voltage_loss_log = []
    current_loss_log = []
    lamda1_log = []
    lamda2_log = []

    for epoch in range(epochs):
        # Training
        model.train()
        # Input normalized power, get normalized voltage prediction
        V_mag_pred_norm, V_ang_pred_norm = model(S_train_norm)
        V_pred_norm = V_mag_pred_norm * torch.exp(1j * V_ang_pred_norm)

        train_loss, volt_loss, curr_loss = pinn_loss(
            V_pred_norm,
            V_true_train_norm,
            Ybus,
            V_norm_params,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
        )

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            V_mag_val_norm, V_ang_val_norm = model(S_val_norm)
            V_val_pred_norm = V_mag_val_norm * torch.exp(1j * V_ang_val_norm)

            val_loss, val_volt_loss, val_curr_loss = pinn_loss(
                V_val_pred_norm,
                V_true_val_norm,
                Ybus,
                V_norm_params,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
            )

        # Adjust learning rate
        scheduler.step(val_loss)

        # Early stopping check
        early_stopping(val_loss, model)

        # Adjust lambda weights every 1000 epochs
        if (epoch + 1) % adjustment_epochs == 0:
            lambda_1 = max(0.5, lambda_1 - adjustment_step)
            lambda_2 = min(0.5, lambda_2 + adjustment_step)

        # Logging
        if epoch % 50 == 0:
            print(
                f"Epoch {epoch}: Train Loss={train_loss.item():.8f}, Val Loss={val_loss.item():.8f}, "
                f"Voltage Loss={volt_loss.item():.8f}, Current Loss={curr_loss.item():.8f}, "
                f'λ1={lambda_1:.5f}, λ2={lambda_2:.5f}, LR={optimizer.param_groups[0]["lr"]:.8f}'
            )

        if epoch % 500 == 0 or epoch == epochs - 1:
            epoch_log.append(epoch)
            train_loss_log.append(train_loss.item())
            val_loss_log.append(val_loss.item())
            voltage_loss_log.append(volt_loss.item())
            current_loss_log.append(curr_loss.item())
            lamda1_log.append(lambda_1)
            lamda2_log.append(lambda_2)

        # Check early stopping
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    # Load the best model for final evaluation
    model.load_state_dict(torch.load(early_stopping.path))
    model.eval()

    # Evaluate on test set
    with torch.no_grad():
        V_mag_test_norm, V_ang_test_norm = model(S_test_norm)
        V_test_pred_norm = V_mag_test_norm * torch.exp(1j * V_ang_test_norm)

        test_loss, test_volt_loss, test_curr_loss = pinn_loss(
            V_test_pred_norm,
            V_true_test_norm,
            Ybus,
            V_norm_params,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
        )

    print(f"\nTest set evaluation:")
    print(f"Total Loss: {test_loss.item():.8f}")
    print(f"Voltage Loss (normalized): {test_volt_loss.item():.8f}")
    print(f"Current Loss (normalized): {test_curr_loss.item():.8f}")

    # Save final model with all important parameters
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalization_params": {
                "V_min_mag": V_min_mag,
                "V_max_mag": V_max_mag,
                "S_min_mag": S_min_mag,
                "S_max_mag": S_max_mag,
            },
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
        },
        f"{results_dir}/final_model.pt",
    )
    print("Final model and normalization parameters saved")

    # Save logs to CSV
    logs = pd.DataFrame(
        {
            "Epoch": epoch_log,
            "Train Loss": train_loss_log,
            "Validation Loss": val_loss_log,
            "Voltage Loss": voltage_loss_log,
            "Current Loss": current_loss_log,
            "Lambda 1": lamda1_log,
            "Lambda 2": lamda2_log,
        }
    )
    logs.to_csv(f"{results_dir}/training_logs.csv", index=False)

    # Plot the error curves
    plt.figure(figsize=(12, 6))

    # Training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epoch_log, train_loss_log, label="Train Loss", linewidth=2)
    plt.plot(epoch_log, val_loss_log, label="Validation Loss", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Component losses
    plt.subplot(1, 2, 2)
    plt.plot(epoch_log, voltage_loss_log, label="Voltage MSE", linewidth=2)
    plt.plot(epoch_log, current_loss_log, label="Current MAE (normalized)", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Component Losses")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_curves.png")
    plt.show()

    # Convert final predictions back to original scale and compare
    V_test_pred_denorm = denormalize_complex_position_wise(
        V_test_pred_norm, *V_norm_params
    )

    # Print comparison for first few buses from test set
    print("\nTest Set Comparison (voltage magnitudes in original scale):")
    print("Bus\t| Predicted\t| True\t| Error %")
    print("-" * 60)
    for i in range(min(5, num_buses)):
        pred_mag = V_test_pred_denorm[0, i].abs().item()
        true_mag = V_true_test[0, i].abs().item()
        error_pct = abs(pred_mag - true_mag) / true_mag * 100
        print(f"{i}\t| {pred_mag:.6f}\t| {true_mag:.6f}\t| {error_pct:.2f}%")

    # Calculate overall test error metrics
    test_voltage_mae = torch.mean(torch.abs(V_test_pred_denorm - V_true_test))
    test_voltage_mape = (
        torch.mean(torch.abs((V_test_pred_denorm - V_true_test) / V_true_test)) * 100
    )

    print(f"\nTest Set Overall Metrics:")
    print(f"Voltage MAE: {test_voltage_mae.item():.6f}")
    print(f"Voltage MAPE: {test_voltage_mape.item():.2f}%")
