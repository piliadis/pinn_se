"""
Currents anti gia powers sto loss
opws edw [1] Falas S et al. Networks for Accelerating Power System State Estimation 2023. https://doi.org/10.48550/arXiv.2310.03088.
                                    Note: to voltage einai MSE enw to current MAE
san to 6 alla me gradually decreasing lambda_1 and increasing lambda_2
0.02 Final Total Loss: 0.00092534 Final Voltage Loss: 0.00051598 Final Current Loss: 0.00153938
0.05 Final Total Loss: 0.00584399 Final Voltage Loss: 0.00092281 Final Current Loss: 0.00584399
0.10 Final Total Loss: 0.00491255 Final Voltage Loss: 0.00235970 Final Current Loss: 0.00491255
0.15 Final Total Loss: 0.00479331 Final Voltage Loss: 0.00773197 Final Current Loss: 0.00479331
0.20 Final Total Loss: 0.00158402 Final Voltage Loss: 0.27944842 Final Current Loss: 0.00158402
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Data loading (explicitly normalized)
def load_data_with_currents(directory, s_base=10):
    P = -pd.read_csv(f"{directory}/real_S.csv", header=None).values / s_base
    Q = -pd.read_csv(f"{directory}/imag_S.csv", header=None).values / s_base
    V_mag = pd.read_csv(f"{directory}/mag_V.csv", header=None).values
    V_ang = pd.read_csv(f"{directory}/ang_V.csv", header=None).values
    Y_real = pd.read_csv(f"{directory}/real_Y.csv", header=None).values
    Y_imag = pd.read_csv(f"{directory}/imag_Y.csv", header=None).values

    Ybus = torch.tensor(Y_real + 1j * Y_imag, dtype=torch.cfloat, device=device)
    V_true = torch.tensor(V_mag * np.exp(1j * V_ang), dtype=torch.cfloat, device=device)
    I_true = torch.matmul(Ybus, V_true.T).T

    S = torch.tensor(P + 1j * Q, dtype=torch.cfloat, device=device)

    return S, V_true, I_true, Ybus


# Model without overly constrained activations
class FlexiblePINN(nn.Module):
    def __init__(self, num_buses):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2 * num_buses, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 2 * num_buses),
        )

    def forward(self, S):
        x = torch.cat([S.real, S.imag], dim=1)
        out = self.model(x)
        V_mag = F.softplus(out[:, :num_buses])
        V_ang = out[:, num_buses:]
        return V_mag, V_ang


def pinn_loss(V_pred, V_true, I_true, Ybus, lambda_1=1.0, lambda_2=1.0):
    # Predicted current injections
    I_pred = torch.matmul(Ybus, V_pred.T).T

    # Voltage MSE Loss (u)
    voltage_mse = torch.mean(torch.abs(V_pred - V_true) ** 2)
    voltage_mse_norm = voltage_mse / torch.max(torch.abs(V_pred - V_true) ** 2)

    # Current MAE Loss (f)
    current_mae = torch.mean(torch.abs(I_pred - I_true))
    current_mae_norm = current_mae / torch.max(torch.abs(I_pred - I_true))

    # Combined loss
    total_loss = lambda_1 * voltage_mse_norm + lambda_2 * current_mae_norm

    return total_loss, voltage_mse_norm, current_mae_norm


if __name__ == "__main__":
    torch.manual_seed(42)
    directory = "data/33bus2"
    S, V_true, I_true, Ybus = load_data_with_currents(directory)

    num_samples, num_buses = S.shape
    model = FlexiblePINN(num_buses).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, patience=500, factor=0.5
    # )

    epochs = 10000

    # Initial weights for loss terms
    lambda_1, lambda_2 = 1.0, 0.0
    adjustment_step = 0.02
    adjustment_epochs = 200

    epoch_log, total_loss_log, voltage_loss_log, current_loss_log = [], [], [], []

    for epoch in range(epochs):
        model.train()
        V_mag_pred, V_ang_pred = model(S)
        V_pred = V_mag_pred * torch.exp(1j * V_ang_pred)

        total_loss, volt_loss, curr_loss = pinn_loss(
            V_pred, V_true, I_true, Ybus, lambda_1=lambda_1, lambda_2=lambda_2
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # scheduler.step(total_loss)

        # Adjust lambda weights every 500 epochs clearly
        if (epoch + 1) % adjustment_epochs == 0:
            lambda_1 = max(0.9, lambda_1 - adjustment_step)
            lambda_2 = min(0.1, lambda_2 + adjustment_step)

        # Clear reporting of λ1 and λ2 values during training
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}: Total Loss={total_loss.item():.8f}, "
                f"Voltage Loss={volt_loss.item():.8f}, Current Loss={curr_loss.item():.8f}, "
                f'λ1={lambda_1:.2f}, λ2={lambda_2:.2f}, LR={optimizer.param_groups[0]["lr"]:.8f}'
            )
            epoch_log.append(epoch)
            total_loss_log.append(total_loss.item())
            voltage_loss_log.append(volt_loss.item())
            current_loss_log.append(curr_loss.item())

    # Final evaluation
    V_mag_final, V_ang_final = model(S)
    V_final_pred = V_mag_final * torch.exp(1j * V_ang_final)
    total_loss, volt_loss, curr_loss = pinn_loss(
        V_final_pred, V_true, I_true, Ybus, lambda_1=lambda_1, lambda_2=lambda_2
    )

    print(f"Final Total Loss: {total_loss.item():.8f}")
    print(f"Final Voltage Loss: {volt_loss.item():.8f}")
    print(f"Final Current Loss: {curr_loss.item():.8f}")

    # Save final model explicitly
    torch.save(model.state_dict(), f"results/33bus/pinn_model{adjustment_step}.pth")
    print("Model saved as pinn_model.pth")

    # Save logs to CSV
    logs = pd.DataFrame(
        {
            "Epoch": epoch_log,
            "Total Loss": total_loss_log,
            "Voltage Loss": voltage_loss_log,
            "Current Loss": current_loss_log,
        }
    )
    logs.to_csv(f"results/33bus/loss_logs{adjustment_step}.csv", index=False)

    # Plot the error curves
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_log, total_loss_log, label="Total Loss", linewidth=2)
    plt.plot(
        epoch_log, voltage_loss_log, label="Normalized Voltage MSE (u)", linewidth=2
    )
    plt.plot(
        epoch_log, current_loss_log, label="Normalized Current MAE (f)", linewidth=2
    )
    plt.xlabel("Epochs")
    plt.ylabel("Normalized Error")
    plt.title("Voltage and Current Errors During Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/33bus/error_curves{adjustment_step}.png")
    plt.show()

    print(V_mag_final[0])
    print(V_true.real[0])

    breakpoint()
