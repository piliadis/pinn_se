import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json
from sklearn.model_selection import train_test_split

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
    "network": "34Bus",
    "epochs": 2000,
    "learning_rate": 0.01,
    "scheduler_patience": 20,
    "scheduler_factor": 0.5,
    "lambda_1_init": 1,
    "lambda_2_init": 0,
    "adjustment_step": 0,
    "adjustment_epochs": 500,
    "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "seed": 42,
    "test_split": 0.2,  # 80/20 train-test split
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


def split_data(S, V_true, I_true, test_split, random_state):
    """Split data into train and test sets"""
    num_samples = S.shape[0]
    indices = torch.arange(num_samples)

    # Convert to numpy for sklearn split
    indices_np = indices.numpy()
    train_idx, test_idx = train_test_split(
        indices_np, test_size=test_split, random_state=random_state, shuffle=True
    )

    # Convert back to torch tensors
    train_idx = torch.from_numpy(train_idx)
    test_idx = torch.from_numpy(test_idx)

    # Split the data
    S_train = S[train_idx]
    V_train = V_true[train_idx]
    I_train = I_true[train_idx]

    S_test = S[test_idx]
    V_test = V_true[test_idx]
    I_test = I_true[test_idx]

    return (S_train, V_train, I_train), (S_test, V_test, I_test), (train_idx, test_idx)


def train(
    model,
    optimizer,
    scheduler,
    S_train_norm,
    V_train,
    V_train_norm,
    I_train,
    Ybus,
    V_norm_params,
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
        V_re_pred_norm, V_im_pred_norm = model(S_train_norm)
        V_pred_norm = torch.complex(V_re_pred_norm, V_im_pred_norm)

        volt_loss_norm = torch.mean(torch.abs(V_pred_norm - V_train_norm) ** 2)

        V_pred_phys = zscore_denormalize_complex(V_pred_norm, *V_norm_params)
        volt_loss = torch.mean(torch.abs(V_pred_phys - V_train) ** 2)

        I_pred = torch.matmul(Ybus, V_pred_phys.T).T
        curr_loss = torch.mean(torch.abs(I_pred - I_train))

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


def evaluate_model(
    model, S_test, V_test, I_test, Ybus, S_norm_params, V_norm_params, device
):
    """Evaluate model on test set"""
    model.eval()

    with torch.no_grad():
        # Normalize test inputs
        S_test_norm, *_ = zscore_normalize_complex(S_test, *S_norm_params)
        V_test_norm, *_ = zscore_normalize_complex(V_test, *V_norm_params)

        # Forward pass
        V_re_pred_norm, V_im_pred_norm = model(S_test_norm)
        V_pred_norm = torch.complex(V_re_pred_norm, V_im_pred_norm)

        # Calculate normalized voltage loss
        volt_loss_norm = torch.mean(torch.abs(V_pred_norm - V_test_norm) ** 2)

        # Denormalize predictions
        V_pred_phys = zscore_denormalize_complex(V_pred_norm, *V_norm_params)

        # Calculate SI voltage loss
        volt_loss = torch.mean(torch.abs(V_pred_phys - V_test) ** 2)

        # Calculate predicted currents and current loss
        I_pred = torch.matmul(Ybus, V_pred_phys.T).T
        curr_loss = torch.mean(torch.abs(I_pred - I_test))

        # Calculate additional metrics
        volt_rmse = torch.sqrt(torch.mean((V_pred_phys - V_test).abs() ** 2))
        volt_mape = torch.mean(torch.abs((V_pred_phys - V_test) / V_test)) * 100
        curr_rmse = torch.sqrt(torch.mean((I_pred - I_test).abs() ** 2))
        curr_mape = torch.mean(torch.abs((I_pred - I_test) / I_test)) * 100

    test_results = {
        "Voltage Loss norm": volt_loss_norm.item(),
        "Voltage Loss SI": volt_loss.item(),
        "Current Loss SI": curr_loss.item(),
        "Voltage RMSE": volt_rmse.item(),
        "Voltage MAPE (%)": volt_mape.item(),
        "Current RMSE": curr_rmse.item(),
        "Current MAPE (%)": curr_mape.item(),
        "Test Samples": S_test.shape[0],
    }

    return test_results, V_pred_phys, I_pred


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

    # Load data
    S, V_true, I_true, Ybus = load_data(data_dir)
    S, V_true, I_true, Ybus = [x.to(device) for x in (S, V_true, I_true, Ybus)]

    # Split data into train and test sets
    (S_train, V_train, I_train), (S_test, V_test, I_test), (train_idx, test_idx) = (
        split_data(S, V_true, I_true, CONFIG["test_split"], CONFIG["seed"])
    )

    print(f"Training samples: {S_train.shape[0]}, Test samples: {S_test.shape[0]}")

    # Normalize training data
    V_train_norm, *V_norm_params = zscore_normalize_complex(V_train)
    S_train_norm, *S_norm_params = zscore_normalize_complex(S_train)

    num_samples, num_buses = S_train.shape
    model, optimizer, scheduler = initialize_model(num_buses, device)

    # Train model
    logs = train(
        model,
        optimizer,
        scheduler,
        S_train_norm,
        V_train,
        V_train_norm,
        I_train,
        Ybus,
        V_norm_params,
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

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results, V_pred_test, I_pred_test = evaluate_model(
        model, S_test, V_test, I_test, Ybus, S_norm_params, V_norm_params, device
    )

    # Print test results
    print("\nTest Set Results:")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    # Save test results to log file
    test_log_path = os.path.join(results_dir, "test_results.json")
    with open(test_log_path, "w") as f:
        json.dump(test_results, f, indent=4)

    # Update CONFIG with final results
    CONFIG["Train Voltage Loss SI"] = logs["Voltage Loss SI"][-1]
    CONFIG["Train Current Loss SI"] = logs["Current Loss SI"][-1]
    CONFIG["Test Results"] = test_results
    CONFIG["Train Samples"] = S_train.shape[0]
    CONFIG["Test Samples"] = S_test.shape[0]

    # Save config
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)

    # Save train/test indices for reproducibility
    torch.save(
        {
            "train_idx": train_idx,
            "test_idx": test_idx,
        },
        os.path.join(results_dir, "data_split_indices.pt"),
    )

    # Final evaluation on full training set (for comparison)
    V_re_pred_norm_final, V_im_pred_norm_final = model(S_train_norm)
    V_pred_norm_final = torch.complex(V_re_pred_norm_final, V_im_pred_norm_final)

    V_pred_phys = zscore_denormalize_complex(V_pred_norm_final, *V_norm_params)
    I_pred = torch.matmul(Ybus, V_pred_phys.T).T

    # Save predictions on training set
    save_complex_tensor(V_pred_phys, "V_pred_train", results_dir)
    save_complex_tensor(I_pred, "I_pred_train", results_dir)

    # Save predictions on test set
    save_complex_tensor(V_pred_test, "V_pred_test", results_dir)
    save_complex_tensor(I_pred_test, "I_pred_test", results_dir)

    print(f"\nResults saved in {results_dir}")
    print(f"Test results log saved at: {test_log_path}")
