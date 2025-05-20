import matplotlib.pyplot as plt
import seaborn as sns
import torch


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


def zscore_normalize_complex(x: torch.Tensor, eps: float = 1e-12):
    """
    Normalise a complex tensor position-wise, with independent μ/σ
    for the real and the imaginary channel.

    Args
    ----
    x   : complex tensor  [batch, …]
    eps : small constant to avoid division by zero

    Returns
    -------
    x_norm         : normalised complex tensor  [batch, …]
    mean_real/imag : real tensors [1, …]  (broadcastable statistics)
    std_real/imag  : real tensors [1, …]
    """
    x_real = x.real
    x_imag = x.imag

    mean_real = x_real.mean(dim=0, keepdim=True)
    std_real = x_real.std(dim=0, keepdim=True, unbiased=False) + eps

    mean_imag = x_imag.mean(dim=0, keepdim=True)
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
