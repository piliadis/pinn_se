import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json

from utils import (
    load_data,
    zscore_normalize_complex,
    zscore_denormalize_complex,
    save_complex_tensor,
)
from model import FlexiblePINN

# at 1st epoch, volt_loss_norm ~ 2 while curr_loss ~7.5m
# volt_loss is normalized, curr_loss is notW
CONFIG = {
    "network": "13Bus",
    "epochs": 2000,
    "learning_rate": 0.01,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
    "lambda_1_init": 1,
    "lambda_2_init": 0.01,
    "adjustment_step": 0,
    "adjustment_epochs": 500,
    "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "seed": 42,
}


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def initialize_model(num_buses, device):
    model = FlexiblePINN(num_buses).to(device).to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=CONFIG["scheduler_patience"],
        factor=CONFIG["scheduler_factor"],
    )
    return model, optimizer, scheduler


def train(
    model,
    optimizer,
    scheduler,
    S_norm,
    V_true,
    V_true_norm,
    I_true,
    Ybus,
    device,
    results_dir,
):
    lambda_1, lambda_2 = CONFIG["lambda_1_init"], CONFIG["lambda_2_init"]

    # Logging
    logs = {
        "Epoch": [],
        "Total Loss": [],
        "Voltage Loss norm": [],
        "Voltage Loss SI": [],
        "Current Loss SI": [],
        "Lambda 1": [],
        "Lambda 2": [],
        "Learning Rate": [],
    }

    for epoch in range(CONFIG["epochs"]):
        model.train()
        V_re_pred_norm, V_im_pred_norm = model(S_norm)
        V_pred_norm = torch.complex(V_re_pred_norm, V_im_pred_norm)

        volt_loss_norm = torch.mean(torch.abs(V_pred_norm - V_true_norm) ** 2)

        V_pred_phys = zscore_denormalize_complex(V_pred_norm, *V_norm_params)
        volt_loss = torch.mean(torch.abs(V_pred_phys - V_true) ** 2)

        I_pred = torch.matmul(Ybus, V_pred_phys.T).T
        curr_loss = torch.mean(torch.abs(I_pred - I_true))

        total_loss = lambda_1 * volt_loss_norm + lambda_2 * curr_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        if epoch % 1000 == 0 and epoch > 0:
            breakpoint()

        # Adjust loss weights
        if (epoch + 1) == CONFIG["adjustment_epochs"]:  # == 0:
            lambda_1 = max(0.5, lambda_1 - CONFIG["adjustment_step"])
            lambda_2 = min(0.5, lambda_2 + CONFIG["adjustment_step"])

        # Logging
        logs["Epoch"].append(epoch)
        logs["Total Loss"].append(total_loss.item())
        logs["Voltage Loss norm"].append(volt_loss_norm.item())
        logs["Voltage Loss SI"].append(volt_loss.item())
        logs["Current Loss SI"].append(curr_loss.item())
        logs["Lambda 1"].append(lambda_1)
        logs["Lambda 2"].append(lambda_2)
        logs["Learning Rate"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:04d}: Total={total_loss.item():.6f}, "
            f"Vnorm={volt_loss_norm.item():.6f}, VSI={volt_loss.item():.6f}, "
            f"ISI={curr_loss.item():.6f}, λ1={lambda_1:.3f}, λ2={lambda_2:.3f}, "
            f"LR={optimizer.param_groups[0]['lr']:.8f}"
        )

    return logs


def plot_logs(logs, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(logs["Epoch"][1:], logs["Total Loss"][1:], label="Total Loss")
    plt.plot(
        logs["Epoch"][1:], logs["Voltage Loss norm"][1:], label="Voltage MSE (norm)"
    )
    plt.plot(logs["Epoch"][1:], logs["Current Loss SI"][1:], label="Current MAE (SI)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(CONFIG["seed"])
    device = setup_device()

    network = CONFIG["network"]
    data_dir = f"data/{network}"
    results_dir = f"results/{network}/run_{CONFIG['run_id']}"

    S, V_true, I_true, Ybus = load_data(data_dir)
    S, V_true, I_true, Ybus = [x.to(device) for x in (S, V_true, I_true, Ybus)]

    V_true_norm, *V_norm_params = zscore_normalize_complex(V_true)
    # I_true_norm, *_ = zscore_normalize_complex(I_true)
    S_norm, *S_norm_params = zscore_normalize_complex(S)

    num_samples, num_buses = S.shape
    model, optimizer, scheduler = initialize_model(num_buses, device)

    logs = train(
        model,
        optimizer,
        scheduler,
        S_norm,
        V_true,
        V_true_norm,
        I_true,
        Ybus,
        device,
        results_dir,
    )

    os.makedirs(results_dir, exist_ok=True)

    # Save model and logs
    torch.save(model.state_dict(), os.path.join(results_dir, "model.pth"))
    normalization_params = {
        "V_mu_re": V_norm_params[0].cpu(),
        "V_sigma_re": V_norm_params[1].cpu(),
        "V_mu_im": V_norm_params[2].cpu(),
        "V_sigma_im": V_norm_params[3].cpu(),
        "S_mu_re": S_norm_params[0].cpu(),
        "S_sigma_re": S_norm_params[1].cpu(),
        "S_mu_im": S_norm_params[2].cpu(),
        "S_sigma_im": S_norm_params[3].cpu(),
    }
    torch.save(
        normalization_params, os.path.join(results_dir, "normalization_params.pt")
    )

    pd.DataFrame(logs).to_csv(
        os.path.join(results_dir, "training_logs.csv"), index=False
    )
    plot_logs(logs, os.path.join(results_dir, "loss_plot.png"))

    CONFIG["Voltage Loss SI"] = logs["Voltage Loss SI"][-1]
    CONFIG["Current Loss SI"] = logs["Current Loss SI"][-1]

    # Save config
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)

    # Final evaluation
    V_re_pred_norm_final, V_im_pred_norm_final = model(S_norm)
    V_pred_norm_final = torch.complex(V_re_pred_norm_final, V_im_pred_norm_final)

    V_pred_phys = zscore_denormalize_complex(V_pred_norm_final, *V_norm_params)
    I_pred = torch.matmul(Ybus, V_pred_phys.T).T

    save_complex_tensor(V_pred_phys, "V_pred", results_dir)
    save_complex_tensor(I_pred, "I_pred", results_dir)

    print(f"Results saved in {results_dir}")
