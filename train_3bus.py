import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from utils import zscore_normalize_complex, zscore_denormalize_complex

from model import FlexiblePINN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_data(directory):
    P = torch.tensor(pd.read_csv(f"{directory}/S_real.csv").values, dtype=torch.float64)
    Q = torch.tensor(pd.read_csv(f"{directory}/S_imag.csv").values, dtype=torch.float64)
    V_real = torch.tensor(
        pd.read_csv(f"{directory}/V_real.csv").values, dtype=torch.float64
    )
    V_imag = torch.tensor(
        pd.read_csv(f"{directory}/V_imag.csv").values, dtype=torch.float64
    )
    Y_real = torch.tensor(
        pd.read_csv(f"{directory}/Y_real.csv", header=None).values, dtype=torch.float64
    )
    Y_imag = torch.tensor(
        pd.read_csv(f"{directory}/Y_imag.csv", header=None).values, dtype=torch.float64
    )
    I_real = torch.tensor(
        pd.read_csv(f"{directory}/I_real.csv").values, dtype=torch.float64
    )
    I_imag = torch.tensor(
        pd.read_csv(f"{directory}/I_imag.csv").values, dtype=torch.float64
    )

    S = torch.complex(P, Q)  # shape: [N, buses]
    V_true = torch.complex(V_real, V_imag)  # shape: [N, buses]
    I_true = torch.complex(I_real, I_imag)  # shape: [N, buses]
    Ybus = torch.complex(Y_real, Y_imag)  # shape: [buses, buses]

    # ~~~~~~~~~~~~~~~  check data ~~~~~~~~~~~~~~~
    # I_calc = torch.matmul(V_true, Ybus)  # shape: [N, buses]
    I_calc = torch.matmul(Ybus, V_true.T).T  # NOTE: to idio
    err1 = torch.mean(torch.abs(I_true - I_calc), axis=1)  # shape: [N]
    assert torch.all(
        err1 < 1e-1
    ), f"Mismatches found! Max error = {torch.max(err1):.8f}"
    S_calc = V_true * torch.conj(
        I_calc
    )  # element-wise complex power, shape: [N, buses]
    err = torch.mean(torch.abs(S - S_calc), dim=1)  # shape: [N]
    assert torch.all(
        err < 1e-1
    ), f"Mismatches found! Max error = {torch.max(err).item():.8f}"

    # # prakseis pou kanei to paper
    # I_calc = torch.matmul(torch.conj(Ybus), V_true.T).T # LATHOS
    I_true = torch.conj(S / V_true)  # SWSTO
    breakpoint()

    return S, V_true, I_true, Ybus


def pinn_loss(V_pred, V_true, I_true, Ybus, lambda_1=1.0, lambda_2=1.0, epoch=50000):
    # Voltage MSE Loss (u)
    voltage_mse = torch.mean(torch.abs(V_pred - V_true) ** 2)
    # voltage_max = torch.max(torch.abs(V_pred - V_true))

    # Current MAE Loss (f)
    I_pred = torch.matmul(Ybus, V_pred.T).T
    current_mae = torch.mean(torch.abs(I_pred - I_true))
    # current_max = torch.max(torch.abs(I_pred - I_true))

    if epoch == 4000:
        breakpoint()

    # Combined loss
    total_loss = lambda_1 * voltage_mse + lambda_2 * current_mae
    # total_loss = (
    #     lambda_1 * voltage_mse / voltage_max + lambda_2 * current_mae / current_max
    # )

    return total_loss, voltage_mse, current_mae


if __name__ == "__main__":
    torch.manual_seed(42)

    network = "3Bus"

    data_dir = "data/" + network
    results_dir = "results/" + network
    os.makedirs(results_dir, exist_ok=True)

    S, V_true, I_true, Ybus = load_data(data_dir)
    S = S.to(device)
    V_true = V_true.to(device)
    I_true = I_true.to(device)
    Ybus = Ybus.to(device)

    V_true_norm, V_mu_re, V_sigma_re, V_mu_im, V_sigma_im = zscore_normalize_complex(
        V_true
    )
    I_true_norm, I_mu_re, I_sigma_re, I_mu_im, I_sigma_im = zscore_normalize_complex(
        I_true
    )
    S_norm, S_mu_re, S_sigma_re, S_mu_im, S_sigma_im = zscore_normalize_complex(S)
    V_true_norm = V_true_norm.to(device)
    I_true_norm = I_true_norm.to(device)
    S_norm = S_norm.to(device)

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
    adjustment_epochs = 200

    epoch_log, total_loss_log, voltage_loss_log, current_loss_log = [], [], [], []

    past_losses = []
    for epoch in range(epochs):
        model.train()
        V_re_pred_norm, V_im_pred_norm = model(S_norm)  # both real tensors
        V_pred_norm = torch.complex(V_re_pred_norm, V_im_pred_norm)

        # Voltage loss in the normalised domain
        volt_loss = torch.mean(torch.abs(V_pred_norm - V_true_norm) ** 2)

        # De-normalise before applying the physics law
        V_pred_phys = zscore_denormalize_complex(
            V_pred_norm, V_mu_re, V_sigma_re, V_mu_im, V_sigma_im
        )

        I_pred = torch.matmul(Ybus, V_pred_phys.T).T
        # I_true = torch.conj(S / V_pred_phys)  # physics reference

        curr_loss = torch.mean(torch.abs(I_pred - I_true))

        total_loss = lambda_1 * volt_loss + lambda_2 * curr_loss

        if epoch % 500 == 0:
            breakpoint()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        # Adjust lambda weights every 500 epochs clearly
        if (epoch + 1) % adjustment_epochs == 0:
            lambda_1 = max(0.5, lambda_1 - adjustment_step)
            lambda_2 = min(0.5, lambda_2 + adjustment_step)

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
