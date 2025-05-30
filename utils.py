import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import os


def plot_heatmap(A):
    plt.figure(figsize=(8, 6))
    sns.heatmap(A, annot=True, cmap="coolwarm", cbar=True)
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.tight_layout()
    plt.show()


def plot_clean_data(dfs):
    plt.figure(figsize=(12, 6))

    for phase_idx, df in enumerate(dfs):
        P = df[["pslack", "p1", "p2", "p3", "p4", "p5"]].values.T
        for i in range(P.shape[0]):
            plt.plot(
                P[i],
                label=f"P{['slack', '1', '2', '3', '4', '5'][i]} - phase {['A','B','C'][phase_idx]}",
            )

    plt.title("Active Power per Bus and Phase (per unit)")
    plt.xlabel("Time Index")
    plt.ylabel("P [W]")
    plt.legend(ncol=3, loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("enfield/dataset2/clean_data.png")
    plt.show()


def plot_2_heatmaps(A, B, title_A="A", title_B="B"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    sns.heatmap(A, annot=True, cmap="coolwarm", cbar=True, ax=axes[0])
    axes[0].set_title(title_A)
    # axes[0].set_xlabel("Column Index")
    # axes[0].set_ylabel("Row Index")

    sns.heatmap(B, annot=True, cmap="coolwarm", cbar=True, ax=axes[1])
    axes[1].set_title(title_B)
    # axes[1].set_xlabel("Column Index")
    # axes[1].set_ylabel("Row Index")

    plt.tight_layout()
    plt.show()


def zscore_normalize_complex(
    x: torch.Tensor,
    eps: float = 1e-12,
    mean_real: torch.Tensor = None,
    std_real: torch.Tensor = None,
    mean_imag: torch.Tensor = None,
    std_imag: torch.Tensor = None,
):
    """
    Normalize a complex tensor position-wise using z-score normalization.

    If mean/std are provided, they are used directly.
    Otherwise, they are computed from the input tensor.

    Args
    ----
    x         : complex tensor [batch, …]
    eps       : small constant to avoid division by zero
    mean_real : optional real tensor [1, …]
    std_real  : optional real tensor [1, …]
    mean_imag : optional real tensor [1, …]
    std_imag  : optional real tensor [1, …]

    Returns
    -------
    x_norm         : normalized complex tensor [batch, …]
    mean_real/imag : real tensors [1, …]
    std_real/imag  : real tensors [1, …]
    """
    x_real = x.real
    x_imag = x.imag

    if mean_real is None:
        mean_real = x_real.mean(dim=0, keepdim=True)
    if std_real is None:
        std_real = x_real.std(dim=0, keepdim=True, unbiased=False) + eps

    if mean_imag is None:
        mean_imag = x_imag.mean(dim=0, keepdim=True)
    if std_imag is None:
        std_imag = x_imag.std(dim=0, keepdim=True, unbiased=False) + eps

    x_norm = torch.complex(
        (x_real - mean_real) / std_real,
        (x_imag - mean_imag) / std_imag,
    )

    return x_norm, mean_real, std_real, mean_imag, std_imag


def zscore_denormalize_complex(
    x_norm: torch.Tensor,
    mean_real: torch.Tensor,
    std_real: torch.Tensor,
    mean_imag: torch.Tensor,
    std_imag: torch.Tensor,
):
    """
    Inverse transform that reconstructs the original complex values.
    """
    return torch.complex(
        x_norm.real * std_real + mean_real,
        x_norm.imag * std_imag + mean_imag,
    )


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
    # I_calc = torch.matmul(Ybus, V_true.T).T  # NOTE: to idio
    I_calc = torch.conj(S / V_true)
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

    return S, V_true, I_true, Ybus


def save_complex_tensor(tensor: torch.Tensor, filename_prefix: str, output_dir: str):
    """
    Saves a complex tensor in two formats:
    - CSV files with real and imaginary parts separately
    - PyTorch .pt file

    Args:
        tensor (torch.Tensor): Complex tensor of shape [samples, features]
        filename_prefix (str): Base name (e.g. "V_pred_phys")
        output_dir (str): Directory where files are saved
    """
    os.makedirs(output_dir, exist_ok=True)
    tensor = tensor.detach().cpu()

    # Save real/imag CSV
    real_df = pd.DataFrame(
        tensor.real.numpy(),
        columns=[f"{filename_prefix}_re_{i}" for i in range(tensor.shape[1])],
    )
    imag_df = pd.DataFrame(
        tensor.imag.numpy(),
        columns=[f"{filename_prefix}_im_{i}" for i in range(tensor.shape[1])],
    )

    real_df.to_csv(os.path.join(output_dir, f"{filename_prefix}_real.csv"), index=False)
    imag_df.to_csv(os.path.join(output_dir, f"{filename_prefix}_imag.csv"), index=False)

    # Save as .pt file
    torch.save(tensor, os.path.join(output_dir, f"{filename_prefix}.pt"))

    print(
        f"Saved: {filename_prefix}_real.csv, {filename_prefix}_imag.csv, and {filename_prefix}.pt to {output_dir}"
    )
