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


def pinn_loss(V_pred, V_true, I_true, Ybus, lambda_1=1.0, lambda_2=1.0):
    # Voltage MSE Loss (u)
    voltage_mse = torch.mean(torch.abs(V_pred - V_true) ** 2)
    # voltage_mse_norm = voltage_mse / torch.max(torch.abs(V_pred - V_true) ** 2)

    # Current MAE Loss (f)
    I_pred = torch.matmul(Ybus, V_pred.T).T
    current_mae = torch.mean(torch.abs(I_pred - I_true))
    # current_mae_norm = current_mae / torch.max(torch.abs(I_pred - I_true))

    # Combined loss
    total_loss = lambda_1 * voltage_mse + lambda_2 * current_mae
    # total_loss = lambda_1 * voltage_mse_norm + lambda_2 * current_mae_norm

    return total_loss, voltage_mse, current_mae


if __name__ == "__main__":
    torch.manual_seed(42)

    network = "123Bus"

    data_dir = "data/" + network
    results_dir = "results/" + network
    os.makedirs(results_dir, exist_ok=True)

    S, V_true, I_true, Ybus = load_data(data_dir)

    num_samples, num_buses = S.shape
    model = FlexiblePINN(num_buses).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
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
        V_mag_pred, V_ang_pred = model(S)
        V_pred = V_mag_pred + 1j * V_ang_pred

        total_loss, volt_loss, curr_loss = pinn_loss(
            V_pred, V_true, I_true, Ybus, lambda_1=lambda_1, lambda_2=lambda_2
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        # Adjust lambda weights every 500 epochs clearly
        if (epoch + 1) % adjustment_epochs == 0:
            lambda_1 = max(0.5, lambda_1 - adjustment_step)
            lambda_2 = min(0.5, lambda_2 + adjustment_step)

        # Clear reporting of λ1 and λ2 values during training
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}: Total Loss={total_loss.detach().item():.8f}, "
                f"Voltage Loss={volt_loss.item():.8f}, Current Loss={curr_loss.item():.8f}, "
                f'λ1={lambda_1:.2f}, λ2={lambda_2:.2f}, LR={optimizer.param_groups[0]["lr"]:.8f}'
            )
            epoch_log.append(epoch)
            total_loss_log.append(total_loss.item())
            voltage_loss_log.append(volt_loss.item())
            current_loss_log.append(curr_loss.item())

    # Final evaluation
    V_mag_final, V_ang_final = model(S)
    V_final_pred = V_mag_final + 1j * V_ang_final
    total_loss, volt_loss, curr_loss = pinn_loss(
        V_final_pred, V_true, I_true, Ybus, lambda_1=lambda_1, lambda_2=lambda_2
    )

    print(f"Final Total Loss: {total_loss.item():.8f}")
    print(f"Final Voltage Loss: {volt_loss.item():.8f}")
    print(f"Final Current Loss: {curr_loss.item():.8f}")

    # Save final model explicitly
    torch.save(model.state_dict(), f"{results_dir}/pinn_model{adjustment_step}.pth")
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
    logs.to_csv(f"{results_dir}/loss_logs{adjustment_step}.csv", index=False)

    # Plot the error curves
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
    plt.savefig(f"{results_dir}/error_curves{adjustment_step}.png")
    plt.show()

    print(V_mag_final[0])
    print(V_true.real[0])

    breakpoint()
