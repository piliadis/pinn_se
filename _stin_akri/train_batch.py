"""
26/5 kamia veltiwsi. ta apotelesmata einai akrivws idia me to xwris batching
"""

import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from utils import load_data
from model import FlexiblePINN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


if __name__ == "__main__":
    torch.manual_seed(42454)

    network = "13Bus"

    data_dir = "data/" + network
    results_dir = "results/" + network
    os.makedirs(results_dir, exist_ok=True)

    S, V_true, I_true, Ybus = load_data(data_dir)
    S = S.to(device)
    V_true = V_true.to(device)
    I_true = I_true.to(device)
    Ybus = Ybus.to(device)
    dataset = TensorDataset(S, V_true, I_true)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_samples, num_buses = S.shape
    model = FlexiblePINN(num_buses).to(device).to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.1
    )

    epochs = 2000

    # Initial weights for loss terms
    lambda_1, lambda_2 = 1, 0.0
    adjustment_step = 0.0001
    adjustment_epochs = 100000

    epoch_log, total_loss_log, voltage_loss_log, current_loss_log = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss_epoch = 0.0
        for S_batch, V_true_batch, I_true_batch in dataloader:
            S_batch = S_batch.to(device)
            V_true_batch = V_true_batch.to(device)
            I_true_batch = I_true_batch.to(device)

            V_re_pred, V_im_pred = model(S_batch)
            V_pred = torch.complex(V_re_pred, V_im_pred)

            volt_loss = torch.mean(torch.abs(V_pred - V_true_batch) ** 2)

            I_pred = torch.matmul(Ybus, V_pred.T).T

            curr_loss = torch.mean(torch.abs(I_pred - I_true_batch))

            total_loss = lambda_1 * volt_loss + lambda_2 * curr_loss

            # if epoch % 300 == 0:
            #     breakpoint()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_epoch += total_loss.item()

        scheduler.step(total_loss_epoch)

        # Adjust lambda weights
        if (epoch + 1) % adjustment_epochs == 0:
            lambda_1 = max(0.8, lambda_1 - adjustment_step)
            lambda_2 = min(0.2, lambda_2 + adjustment_step)

        print(f"Epoch {epoch}: Total Loss={total_loss_epoch:.8f} ")
        epoch_log.append(epoch)
        total_loss_log.append(total_loss_epoch)
        # voltage_loss_log.append(volt_loss.item())
        # current_loss_log.append(curr_loss.item())

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
        label="Voltage MSE (u)",
        linewidth=2,
    )
    plt.plot(
        epoch_log[1:],
        current_loss_log[1:],
        label="Current MAE (f)",
        linewidth=2,
    )
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Voltage and Current Errors During Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/error_curves{adjustment_step}.png")
    plt.show()

    breakpoint()
