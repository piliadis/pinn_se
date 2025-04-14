import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Correct data loading with proper pu and SI handling
def load_data(directory, s_base=1_000_000):
    P_pu = -pd.read_csv(f"{directory}/real_S.csv").values
    P_pu[0:3] *= -1
    Q_pu = -pd.read_csv(f"{directory}/imag_S.csv").values
    Q_pu[0:3] *= -1
    V_mag_pu = pd.read_csv(f"{directory}/mag_V.csv").values
    V_ang_rad = pd.read_csv(f"{directory}/ang_V_rad.csv").values
    Y_real_pu = pd.read_csv(f"{directory}/real_Y_SI.csv", header=None).values
    Y_imag_pu = pd.read_csv(f"{directory}/imag_Y_SI.csv", header=None).values

    Ybus_pu = torch.tensor(
        Y_real_pu + 1j * Y_imag_pu, dtype=torch.cfloat, device=device
    )
    V_true_pu = torch.tensor(
        V_mag_pu * np.exp(1j * V_ang_rad), dtype=torch.cfloat, device=device
    )
    S_pu = torch.tensor(P_pu + 1j * Q_pu, dtype=torch.cfloat, device=device)

    I_true_pu = torch.matmul(Ybus_pu, V_true_pu.T).T

    return S_pu, V_true_pu, I_true_pu, Ybus_pu


class FlexiblePINN(nn.Module):
    def __init__(self, num_buses):
        super().__init__()
        self.num_buses = num_buses
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
        V_mag = out[:, : self.num_buses]
        V_ang = out[:, self.num_buses :]
        return V_mag, V_ang


# Your original loss function
def pinn_loss(V_pred, V_true, I_true, Ybus, lambda_1=1.0, lambda_2=1.0):
    I_pred = torch.matmul(Ybus, V_pred.T).T

    voltage_mse = torch.mean(torch.abs(V_pred - V_true) ** 2)
    voltage_mse_norm = voltage_mse / torch.max(torch.abs(V_pred - V_true) ** 2)

    current_mae = torch.mean(torch.abs(I_pred - I_true))
    current_mae_norm = current_mae / torch.max(torch.abs(I_pred - I_true))

    total_loss = lambda_1 * voltage_mse_norm + lambda_2 * current_mae_norm

    return total_loss, voltage_mse_norm, current_mae_norm


# Small function to verify physical correctness (I=Y*V)
def check_physical_correctness(V, I, Ybus):
    I_calc = torch.matmul(Ybus, V.T).T
    error_max = torch.max(torch.abs(I - I_calc))
    error_min = torch.min(torch.abs(I - I_calc))
    error = torch.mean(torch.abs(I - I_calc))
    return error_max, error_min


if __name__ == "__main__":
    torch.manual_seed(42)
    directory = "data/13bus"
    S, V_true, I_true, Ybus = load_data(directory)

    for i in range(V_true.shape[0]):
        I_calc = torch.matmul(Ybus, V_true[i])  # V[i]: shape (num_buses,)
        err = torch.mean(torch.abs(I_true[i] - I_calc))
        assert err < 1e-6, f"Mismatch at sample {i}: error={err.item():.8f}"

    breakpoint()

    num_samples, num_buses = S.shape
    model = FlexiblePINN(num_buses).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5000
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

        if (epoch + 1) % adjustment_epochs == 0:
            lambda_1 = max(0.9, lambda_1 - adjustment_step)
            lambda_2 = min(0.1, lambda_2 + adjustment_step)

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}: Total Loss={total_loss.item():.8f}, Voltage Loss={volt_loss.item():.8f}, Current Loss={curr_loss.item():.8f}, λ1={lambda_1:.2f}, λ2={lambda_2:.2f}"
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

    torch.save(model.state_dict(), f"ann/pinn_model{adjustment_step}.pth")
    print("Model saved as pinn_model.pth")

    logs = pd.DataFrame(
        {
            "Epoch": epoch_log,
            "Total Loss": total_loss_log,
            "Voltage Loss": voltage_loss_log,
            "Current Loss": current_loss_log,
        }
    )
    logs.to_csv(f"ann/loss_logs{adjustment_step}.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_log[1:], total_loss_log[1:], label="Total Loss", linewidth=2)
    plt.plot(
        epoch_log[1:],
        voltage_loss_log[1:],
        label="Normalized Voltage MSE (u)",
        linewidth=2,
    )
    plt.plot(
        epoch_log[1:],
        current_loss_log[1:],
        label="Normalized Current MAE (f)",
        linewidth=2,
    )
    plt.xlabel("Epochs")
    plt.ylabel("Normalized Error")
    plt.title("Voltage and Current Errors During Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"ann/error_curves{adjustment_step}.png")
    plt.show()

    print(V_mag_final[0])
    print(V_true.real[0])

    breakpoint()
