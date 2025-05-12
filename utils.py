import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_2_heatmaps(A, B):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    sns.heatmap(A, annot=True, cmap="coolwarm", cbar=True, ax=axes[0])
    # axes[0].set_title("Heatmap of A")
    axes[0].set_xlabel("Column Index")
    axes[0].set_ylabel("Row Index")

    sns.heatmap(B, annot=True, cmap="coolwarm", cbar=True, ax=axes[1])
    # axes[1].set_title("Heatmap of B")
    axes[1].set_xlabel("Column Index")
    axes[1].set_ylabel("Row Index")

    plt.tight_layout()
    plt.show()
