"""
21/5/2025
EINAI SWSTO. apla thelei isws to I error na einai normalized giati to V einai
den einai aplo.. dokimasa ena  Robust normalization using median and IQR (interquartile range)
alla den doulevei. kalytera na minei SI

26/5/2025
xwris norm den kanoume tipota. tromeri veltiwsi. tsekarw pali an mporw na kanw norm ta I
"""

import torch
import pandas as pd
import os
import matplotlib.pyplot as plt

from utils import load_data, zscore_normalize_complex, zscore_denormalize_complex
from model import FlexiblePINN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


if __name__ == "__main__":
    torch.manual_seed(42)

    network = "13Bus"

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
        optimizer, patience=50, factor=0.5
    )

    epochs = 1000

    # Initial weights for loss terms
    lambda_1, lambda_2 = 1, 0
    adjustment_step = 0.00 / 1e7
    adjustment_epochs = 100

    (
        epoch_log,
        total_loss_log,
        voltage_loss_norm_log,
        current_loss_log,
        lambda_1_log,
        lambda_2_log,
        learning_rate_log,
    ) = ([], [], [], [], [], [], [])

    for epoch in range(epochs):
        model.train()
        V_re_pred_norm, V_im_pred_norm = model(S_norm)  # both real tensors
        V_pred_norm = torch.complex(V_re_pred_norm, V_im_pred_norm)

        # Voltage loss in the normalised domain
        volt_loss_norm = torch.mean(torch.abs(V_pred_norm - V_true_norm) ** 2)

        # De-normalise before applying the physics law
        V_pred_phys = zscore_denormalize_complex(
            V_pred_norm, V_mu_re, V_sigma_re, V_mu_im, V_sigma_im
        )

        volt_loss = torch.mean(torch.abs(V_pred_phys - V_true) ** 2)

        I_pred = torch.matmul(Ybus, V_pred_phys.T).T  # NOTE: this is SI!!

        curr_loss = torch.mean(torch.abs(I_pred - I_true))

        total_loss = lambda_1 * volt_loss_norm + lambda_2 * curr_loss
        # at 1st epoch, volt_loss ~ 2 while curr_loss ~7.5m
        # volt_loss is normalized, curr_loss is not

        if epoch % 10000 == 0:
            breakpoint()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        # Adjust lambda weights
        if (epoch + 1) % adjustment_epochs == 0:
            lambda_1 = max(0.5, lambda_1 - adjustment_step)
            lambda_2 = min(0.5, lambda_2 + adjustment_step)

        print(
            f"Epoch {epoch}: Total Loss={total_loss.detach().item():.8f}, "
            f"Voltage Loss norm={volt_loss_norm.item():.8f},Voltage Loss SI={volt_loss.item():.8f}, Current Loss={curr_loss.item():.8f}, "
            f'λ1={lambda_1:.3f}, λ2={lambda_2:.3f}, LR={optimizer.param_groups[0]["lr"]:.8f}'
        )
        epoch_log.append(epoch)
        total_loss_log.append(total_loss.item())
        voltage_loss_norm_log.append(volt_loss_norm.item())
        current_loss_log.append(curr_loss.item())
        lambda_1_log.append(lambda_1)
        lambda_2_log.append(lambda_2)
        learning_rate_log.append(optimizer.param_groups[0]["lr"])

    # Final evaluation
    V_re_pred_norm_final, V_im_pred_norm_final = model(S_norm)
    V_pred_norm_final = torch.complex(V_re_pred_norm_final, V_im_pred_norm_final)

    volt_loss_norm = torch.mean(torch.abs(V_pred_norm_final - V_true_norm) ** 2)

    # De-normalise before applying the physics law
    V_pred_phys = zscore_denormalize_complex(
        V_pred_norm_final, V_mu_re, V_sigma_re, V_mu_im, V_sigma_im
    )

    I_pred = torch.matmul(Ybus, V_pred_phys.T).T  # NOTE: SI!!

    curr_loss = torch.mean(torch.abs(I_pred - I_true))

    total_loss = lambda_1 * volt_loss_norm + lambda_2 * curr_loss

    print(f"Final Total Loss: {total_loss.item():.8f}")
    print(f"Final Voltage Loss norm: {volt_loss_norm.item():.8f}")
    print(f"Final Current Loss: {curr_loss.item():.8f}")

    # Save final model explicitly
    torch.save(model.state_dict(), f"{results_dir}/pinn_model{adjustment_step}.pth")
    print("Model saved as pinn_model.pth")

    # Save logs to CSV
    logs = pd.DataFrame(
        {
            "Epoch": epoch_log,
            "Total Loss": total_loss_log,
            "Voltage Loss norm": voltage_loss_norm_log,
            "Current Loss": current_loss_log,
            "Lambda 1": lambda_1_log,
            "Lambda 2": lambda_2_log,
            "Learning Rate": learning_rate_log,
        }
    )
    logs.to_csv(f"{results_dir}/loss_logs{adjustment_step}.csv", index=False)

    # Plot the error curves
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_log[1:], total_loss_log[1:], label="Total Loss", linewidth=2)
    plt.plot(
        epoch_log[1:],
        voltage_loss_norm_log[1:],
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
