import torch
import pandas as pd
import os
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
import numpy as np

from utils import load_data, save_complex_tensor
from model import FlexiblePINN


CONFIG = {
    "network": "13Bus",
    "epochs": 2000,
    "learning_rate": 0.01,
    "scheduler_patience": 50,
    "scheduler_factor": 0.5,
    "lambda_1_init": 1,
    "lambda_2_init": 0.001,
    "adjustment_step": 0,
    "adjustment_epochs": 500,
    "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "seed": 42,
    "test_split": 0.2,
    "weight_decay": 1e-5,
    "epsilon": 1e-3,  # For robust normalization
}


def robust_normalize_complex(data, epsilon=1e-3, norm_params=None):
    """
    Robust normalization for complex data that prevents explosion

    If norm_params is provided, apply those parameters instead of calculating new ones.
    norm_params should be a tuple (mu_re, sigma_re, mu_im, sigma_im).
    """
    # Separate real and imaginary parts
    data_real = data.real
    data_imag = data.imag

    if norm_params is None:
        # Compute statistics per bus (along batch dimension)
        mu_re = data_real.mean(dim=0, keepdim=True)
        sigma_re = data_real.std(dim=0, keepdim=True)
        mu_im = data_imag.mean(dim=0, keepdim=True)
        sigma_im = data_imag.std(dim=0, keepdim=True)

        # Add epsilon to prevent division by zero and clip very small values
        sigma_re = torch.clamp(sigma_re, min=epsilon * data_real.abs().mean())
        sigma_im = torch.clamp(sigma_im, min=epsilon * data_imag.abs().mean())
    else:
        # Unpack provided normalization parameters
        mu_re, sigma_re, mu_im, sigma_im = norm_params

    # Normalize
    data_norm_real = (data_real - mu_re) / sigma_re
    data_norm_imag = (data_imag - mu_im) / sigma_im

    # Clip extreme values to prevent explosion
    data_norm_real = torch.clamp(data_norm_real, min=-10, max=10)
    data_norm_imag = torch.clamp(data_norm_imag, min=-10, max=10)

    data_norm = torch.complex(data_norm_real, data_norm_imag)

    return data_norm, (mu_re, sigma_re, mu_im, sigma_im)


def robust_denormalize_complex(data_norm, mu_re, sigma_re, mu_im, sigma_im):
    """Denormalize complex data"""
    data_real = data_norm.real * sigma_re + mu_re
    data_imag = data_norm.imag * sigma_im + mu_im
    return torch.complex(data_real, data_imag)


def diagnose_normalization(
    S_train, V_train, S_test, V_test, S_norm_params, V_norm_params
):
    """Check normalization statistics"""
    print("\n=== NORMALIZATION DIAGNOSIS ===")

    # Check sigma values
    print("\nSigma values (should not be too small):")
    print(
        f"S sigma_re - Min: {S_norm_params[1].min():.6f}, Mean: {S_norm_params[1].mean():.6f}"
    )
    print(
        f"S sigma_im - Min: {S_norm_params[3].min():.6f}, Mean: {S_norm_params[3].mean():.6f}"
    )
    print(
        f"V sigma_re - Min: {V_norm_params[1].min():.6f}, Mean: {V_norm_params[1].mean():.6f}"
    )
    print(
        f"V sigma_im - Min: {V_norm_params[3].min():.6f}, Mean: {V_norm_params[3].mean():.6f}"
    )

    # Normalize and check ranges
    S_train_norm = robust_normalize_complex(S_train, epsilon=0)[0]
    V_train_norm = robust_normalize_complex(V_train, epsilon=0)[0]
    S_test_norm = robust_normalize_complex(S_test, epsilon=0)[0]
    V_test_norm = robust_normalize_complex(V_test, epsilon=0)[0]

    print("\nNormalized data ranges (should be roughly -3 to 3):")
    print(
        f"S_train_norm - Min: {S_train_norm.abs().min():.2f}, Max: {S_train_norm.abs().max():.2f}"
    )
    print(
        f"V_train_norm - Min: {V_train_norm.abs().min():.2f}, Max: {V_train_norm.abs().max():.2f}"
    )
    print(
        f"S_test_norm - Min: {S_test_norm.abs().min():.2f}, Max: {S_test_norm.abs().max():.2f}"
    )
    print(
        f"V_test_norm - Min: {V_test_norm.abs().min():.2f}, Max: {V_test_norm.abs().max():.2f}"
    )


def train(
    model,
    optimizer,
    scheduler,
    S_train_norm,
    V_train,
    V_train_norm,
    I_train,
    S_test,
    V_test,
    I_test,
    Ybus,
    S_norm_params,
    V_norm_params,
    device,
    results_dir,
):
    lambda_1, lambda_2 = CONFIG["lambda_1_init"], CONFIG["lambda_2_init"]

    best_test_loss = float("inf")
    best_epoch = 0

    logs = {
        "Epoch": [],
        "Total Loss": [],
        "Voltage Loss norm": [],
        "Voltage Loss SI": [],
        "Current Loss SI": [],
        "Test Voltage Loss SI": [],
        "Test Current Loss SI": [],
    }

    for epoch in range(CONFIG["epochs"]):
        # Training
        model.train()
        V_re_pred_norm, V_im_pred_norm = model(S_train_norm)
        V_pred_norm = torch.complex(V_re_pred_norm, V_im_pred_norm)

        volt_loss_norm = torch.mean(torch.abs(V_pred_norm - V_train_norm) ** 2)

        V_pred_phys = robust_denormalize_complex(V_pred_norm, *V_norm_params)
        volt_loss = torch.mean(torch.abs(V_pred_phys - V_train) ** 2)

        I_pred = torch.matmul(Ybus, V_pred_phys.T).T
        curr_loss = torch.mean(torch.abs(I_pred - I_train))

        total_loss = lambda_1 * volt_loss_norm + lambda_2 * curr_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)

        # Evaluate on test set every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Normalize test inputs using the saved normalization parameters
                S_test_norm, _ = robust_normalize_complex(
                    S_test, norm_params=S_norm_params
                )

                # Forward pass
                V_re_test_norm, V_im_test_norm = model(S_test_norm)
                V_test_pred_norm = torch.complex(V_re_test_norm, V_im_test_norm)

                # Denormalize predictions
                V_test_pred_phys = robust_denormalize_complex(
                    V_test_pred_norm, *V_norm_params
                )
                test_volt_loss = torch.mean(torch.abs(V_test_pred_phys - V_test) ** 2)

                I_test_pred = torch.matmul(Ybus, V_test_pred_phys.T).T
                test_curr_loss = torch.mean(torch.abs(I_test_pred - I_test))

                # Save best model
                test_total_loss = test_volt_loss + test_curr_loss
                if test_total_loss < best_test_loss:
                    best_test_loss = test_total_loss
                    best_epoch = epoch
                    torch.save(
                        model.state_dict(), os.path.join(results_dir, "best_model.pth")
                    )

        # Adjust loss weights
        if (epoch + 1) == CONFIG["adjustment_epochs"]:
            lambda_1 = max(0.5, lambda_1 - CONFIG["adjustment_step"])
            lambda_2 = min(0.5, lambda_2 + CONFIG["adjustment_step"])

        # Logging
        logs["Epoch"].append(epoch)
        logs["Total Loss"].append(total_loss.item())
        logs["Voltage Loss norm"].append(volt_loss_norm.item())
        logs["Voltage Loss SI"].append(volt_loss.item())
        logs["Current Loss SI"].append(curr_loss.item())

        if epoch % 10 == 0:
            logs["Test Voltage Loss SI"].append(test_volt_loss.item())
            logs["Test Current Loss SI"].append(test_curr_loss.item())

            print(
                f"Epoch {epoch:04d}: Total={total_loss.item():.6f}, "
                f"V_norm={volt_loss_norm.item():.6f}, V_SI={volt_loss.item():.6f}, "
                f"I_SI={curr_loss.item():.6f}, "
                f"Test V_SI={test_volt_loss.item():.6f}, "
                f"Test I_SI={test_curr_loss.item():.6f}, "
                f"LR={optimizer.param_groups[0]['lr']:.8f}"
            )

    print(f"\nBest model saved at epoch {best_epoch}")
    return logs


def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    network = CONFIG["network"]
    data_dir = f"data/{network}"
    results_dir = f"results/{network}/run_{CONFIG['run_id']}"
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    S, V_true, I_true, Ybus = load_data(data_dir)
    S, V_true, I_true, Ybus = [x.to(device) for x in (S, V_true, I_true, Ybus)]

    # Split data
    indices = torch.arange(S.shape[0])
    train_idx, test_idx = train_test_split(
        indices.numpy(),
        test_size=CONFIG["test_split"],
        random_state=CONFIG["seed"],
        shuffle=True,
    )
    train_idx = torch.from_numpy(train_idx)
    test_idx = torch.from_numpy(test_idx)

    S_train = S[train_idx]
    V_train = V_true[train_idx]
    I_train = I_true[train_idx]

    S_test = S[test_idx]
    V_test = V_true[test_idx]
    I_test = I_true[test_idx]

    print(f"\nTraining samples: {S_train.shape[0]}, Test samples: {S_test.shape[0]}")

    # Normalize with robust method
    print("\nApplying robust normalization...")
    V_train_norm, V_norm_params = robust_normalize_complex(
        V_train, epsilon=CONFIG["epsilon"]
    )
    S_train_norm, S_norm_params = robust_normalize_complex(
        S_train, epsilon=CONFIG["epsilon"]
    )

    # Diagnose normalization
    diagnose_normalization(
        S_train, V_train, S_test, V_test, S_norm_params, V_norm_params
    )

    # Initialize model
    num_buses = S_train.shape[1]
    model = FlexiblePINN(num_buses).to(device).to(torch.float64)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=CONFIG["scheduler_patience"],
        factor=CONFIG["scheduler_factor"],
    )

    # Train
    logs = train(
        model,
        optimizer,
        scheduler,
        S_train_norm,
        V_train,
        V_train_norm,
        I_train,
        S_test,
        V_test,
        I_test,
        Ybus,
        S_norm_params,
        V_norm_params,
        device,
        results_dir,
    )

    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth")))

    # Final evaluation
    model.eval()
    with torch.no_grad():
        S_test_norm, _ = robust_normalize_complex(S_test, norm_params=S_norm_params)
        V_test_norm, _ = robust_normalize_complex(V_test, norm_params=V_norm_params)

        V_re_pred_norm, V_im_pred_norm = model(S_test_norm)
        V_pred_norm = torch.complex(V_re_pred_norm, V_im_pred_norm)

        volt_loss_norm = torch.mean(torch.abs(V_pred_norm - V_test_norm) ** 2)

        V_pred_phys = robust_denormalize_complex(V_pred_norm, *V_norm_params)
        volt_mse = torch.mean(torch.abs(V_pred_phys - V_test))

        I_pred = torch.matmul(Ybus, V_pred_phys.T).T
        curr_mse = torch.mean(torch.abs(I_pred - I_test))

        volt_rmse = torch.sqrt(torch.mean((V_pred_phys - V_test).abs() ** 2))
        curr_rmse = torch.sqrt(torch.mean((I_pred - I_test).abs() ** 2))

        V_test_clamped = torch.complex(
            V_test.real.clamp(min=1e-6), V_test.imag.clamp(min=1e-6)
        )
        volt_mape = torch.mean(torch.abs((V_pred_phys - V_test) / V_test_clamped))
        I_test_clamped = torch.complex(
            I_test.real.clamp(min=1e-6), I_test.imag.clamp(min=1e-6)
        )
        curr_mape = torch.mean(torch.abs((I_pred - I_test) / I_test_clamped))

    test_results = {
        "Voltage Loss norm": volt_loss_norm.item(),
        "Voltage MAE": volt_mse.item(),
        "Current MAE": curr_mse.item(),
        "Voltage RMSE": volt_rmse.item(),
        "Current RMSE": curr_rmse.item(),
        "Voltage MAPE": volt_mape.item(),
        "Current MAPE": curr_mape.item(),
        "Test Samples": S_test.shape[0],
    }

    print("\n=== FINAL TEST RESULTS ===")
    for key, value in test_results.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")

    # Save everything
    torch.save(model.state_dict(), os.path.join(results_dir, "final_model.pth"))
    torch.save(
        {
            "V_norm_params": V_norm_params,
            "S_norm_params": S_norm_params,
        },
        os.path.join(results_dir, "normalization_params.pt"),
    )
    # breakpoint()
    lol1 = logs["Test Voltage Loss SI"][0]
    lol2 = logs["Test Current Loss SI"][0]
    logs["Test Voltage Loss SI"] = list(np.zeros(len(logs["Epoch"])))
    logs["Test Current Loss SI"] = list(np.zeros(len(logs["Epoch"])))
    logs["Test Voltage Loss SI"][-1] = lol1
    logs["Test Current Loss SI"][-1] = lol2
    pd.DataFrame(logs).to_csv(
        os.path.join(results_dir, "training_logs.csv"), index=False
    )

    with open(os.path.join(results_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=4)

    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)

    save_complex_tensor(V_pred_phys, "V_pred_test", results_dir)
    save_complex_tensor(I_pred, "I_pred_test", results_dir)

    print(f"\nResults saved in {results_dir}")


if __name__ == "__main__":
    main()
