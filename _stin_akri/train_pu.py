"""
21/5/2025
EINAI SWSTO. apla thelei isws to I error na einai normalized giati to V einai
den einai aplo.. dokimasa ena  Robust normalization using median and IQR (interquartile range)
alla den doulevei. kalytera na minei SI

vazw apla to load_data_pu. evala kai curr_loss / 1e6
ola ta alla to ido
"""

import torch
import pandas as pd
import os
import matplotlib.pyplot as plt

from utils import load_data
from model import FlexiblePINN


def load_data_pu(directory):
    P = torch.tensor(
        pd.read_csv(f"{directory}/S_real_pu.csv").values, dtype=torch.float64
    )
    Q = torch.tensor(
        pd.read_csv(f"{directory}/S_imag_pu.csv").values, dtype=torch.float64
    )
    V_real = torch.tensor(
        pd.read_csv(f"{directory}/V_real_pu.csv").values, dtype=torch.float64
    )
    V_imag = torch.tensor(
        pd.read_csv(f"{directory}/V_imag_pu.csv").values, dtype=torch.float64
    )
    Y_real = torch.tensor(
        pd.read_csv(f"{directory}/Y_real_pu.csv", header=None).values,
        dtype=torch.float64,
    )
    Y_imag = torch.tensor(
        pd.read_csv(f"{directory}/Y_imag_pu.csv", header=None).values,
        dtype=torch.float64,
    )
    I_real = torch.tensor(
        pd.read_csv(f"{directory}/I_real_pu.csv").values, dtype=torch.float64
    )
    I_imag = torch.tensor(
        pd.read_csv(f"{directory}/I_imag_pu.csv").values, dtype=torch.float64
    )

    S = torch.complex(P, Q)  # shape: [N, buses]
    V_true = torch.complex(V_real, V_imag)  # shape: [N, buses]
    I_true = torch.complex(I_real, I_imag)  # shape: [N, buses]
    Ybus = torch.complex(Y_real, Y_imag)  # shape: [buses, buses]

    # ~~~~~~~~~~~~~~~  check data ~~~~~~~~~~~~~~~
    # I_calc = torch.matmul(V_true, Ybus)  # shape: [N, buses]
    # I_calc = torch.matmul(Ybus, V_true.T).T  # NOTE: to idio
    I_calc = torch.conj(S / V_true)
    err1 = torch.mean(torch.abs(I_true - I_calc), axis=1)  # shape: [N]
    assert torch.all(
        err1 < 1e-10
    ), f"Mismatches found! Max error = {torch.max(err1):.8f}"
    S_calc = V_true * torch.conj(
        I_calc
    )  # element-wise complex power, shape: [N, buses]
    err = torch.mean(torch.abs(S - S_calc), dim=1)  # shape: [N]
    assert torch.all(
        err < 1e-10
    ), f"Mismatches found! Max error = {torch.max(err).item():.8f}"

    return S, V_true, I_true, Ybus


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


if __name__ == "__main__":
    # torch.manual_seed(42)

    network = "13Bus"

    data_dir = "data/" + network
    results_dir = "results/" + network
    os.makedirs(results_dir, exist_ok=True)

    S, V_true, I_true, Ybus = load_data_pu(data_dir)
    S = S.to(device)
    V_true = V_true.to(device)
    I_true = I_true.to(device)
    Ybus = Ybus.to(device)

    num_samples, num_buses = S.shape
    model = FlexiblePINN(num_buses).to(device).to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=200, factor=0.5
    )

    epochs = 20000

    # Initial weights for loss terms
    lambda_1, lambda_2 = 1, 0
    adjustment_step = 0.05
    adjustment_epochs = 2000

    epoch_log, total_loss_log, voltage_loss_log, current_loss_log = [], [], [], []

    for epoch in range(epochs):
        model.train()
        V_re_pred, V_im_pred = model(S)  # both real tensors
        V_pred = torch.complex(V_re_pred, V_im_pred)

        # Voltage loss in the normalised domain
        volt_loss = torch.mean(torch.abs(V_pred - V_true) ** 2)

        I_pred = torch.matmul(Ybus, V_pred.T).T  # NOTE: ayto einai SI!!

        curr_loss = torch.mean(torch.abs(I_pred - I_true))

        total_loss = lambda_1 * volt_loss + lambda_2 * curr_loss / 1e6

        if epoch % 5000 == 0:
            breakpoint()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        # Adjust lambda weights
        # if (epoch + 1) % adjustment_epochs == 0:
        #     lambda_1 = max(0.8, lambda_1 - adjustment_step)
        #     lambda_2 = min(0.2, lambda_2 + adjustment_step)

        print(
            f"Epoch {epoch}: Total Loss={total_loss.detach().item():.8f}, "
            f"Voltage Loss={volt_loss.item():.8f}, Current Loss={curr_loss.item():.8f}, "
            f'λ1={lambda_1:.3f}, λ2={lambda_2:.3f}, LR={optimizer.param_groups[0]["lr"]:.8f}'
        )
        epoch_log.append(epoch)
        total_loss_log.append(total_loss.item())
        voltage_loss_log.append(volt_loss.item())
        current_loss_log.append(curr_loss.item())

    # Final evaluation
    V_re_pred_final, V_im_pred_final = model(S)
    V_pred_final = torch.complex(V_re_pred_final, V_im_pred_final)

    volt_loss = torch.mean(torch.abs(V_pred_final - V_true) ** 2)

    I_pred = torch.matmul(Ybus, V_pred_final.T).T  # NOTE: ayto einai SI!!

    curr_loss = torch.mean(torch.abs(I_pred - I_true))

    total_loss = lambda_1 * volt_loss + lambda_2 * curr_loss

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

    breakpoint()
