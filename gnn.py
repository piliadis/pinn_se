import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric.data as pyg_data
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader


# Load data and create PyG graph dataset
def load_33bus_data(directory, s_base=10):
    P = -pd.read_csv(f"{directory}/real_S.csv", header=None).values / s_base
    Q = -pd.read_csv(f"{directory}/imag_S.csv", header=None).values / s_base
    V_mag = pd.read_csv(f"{directory}/mag_V.csv", header=None).values
    V_ang = pd.read_csv(f"{directory}/ang_V.csv", header=None).values
    Y_real = pd.read_csv(f"{directory}/real_Y.csv", header=None).values
    Y_imag = pd.read_csv(f"{directory}/imag_Y.csv", header=None).values

    Ybus = torch.tensor(Y_real + 1j * Y_imag, dtype=torch.cfloat)
    V_true = torch.tensor(V_mag * np.exp(1j * V_ang), dtype=torch.cfloat)
    I_true = torch.matmul(Ybus, V_true.T).T

    G = torch.tensor(Y_real, dtype=torch.float32)
    B = torch.tensor(Y_imag, dtype=torch.float32)
    edge_index = torch.nonzero(G, as_tuple=False).t()
    edge_attr = torch.stack(
        [G[edge_index[0], edge_index[1]], B[edge_index[0], edge_index[1]]], dim=1
    )

    dataset = []
    for i in range(P.shape[0]):
        x = torch.tensor(np.stack([P[i], Q[i]]).T, dtype=torch.float32)
        y = torch.tensor(np.stack([V_mag[i], V_ang[i]]).T, dtype=torch.float32)
        v = torch.tensor(V_true[i], dtype=torch.cfloat)
        i_true = torch.matmul(Ybus, v)

        data = pyg_data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            V_true=v,
            I_true=i_true,
        )
        dataset.append(data)

    return dataset, Ybus


# GNN model
class GNNPINN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()

        # Edge MLPs for each layer
        self.edge_mlp1 = nn.Sequential(nn.Linear(2, input_dim * hidden_dim), nn.ReLU())
        self.edge_mlp2 = nn.Sequential(nn.Linear(2, hidden_dim * hidden_dim), nn.ReLU())
        self.edge_mlp3 = nn.Sequential(nn.Linear(2, hidden_dim * output_dim), nn.ReLU())

        self.conv1 = pyg_nn.NNConv(input_dim, hidden_dim, self.edge_mlp1, aggr="mean")
        self.conv2 = pyg_nn.NNConv(hidden_dim, hidden_dim, self.edge_mlp2, aggr="mean")
        self.conv3 = pyg_nn.NNConv(hidden_dim, output_dim, self.edge_mlp3, aggr="mean")

        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.relu(self.conv1(x, edge_index, edge_attr))
        x = self.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x


# PINN loss (voltage + current)
def pinn_loss(V_pred, V_true, I_true, Ybus, lambda_1=1.0, lambda_2=1.0):
    I_pred = torch.matmul(Ybus, V_pred.T).T
    voltage_mse = torch.mean(torch.abs(V_pred - V_true) ** 2)
    voltage_mse_norm = voltage_mse / torch.max(torch.abs(V_pred - V_true) ** 2)
    current_mae = torch.mean(torch.abs(I_pred - I_true))
    current_mae_norm = current_mae / torch.max(torch.abs(I_pred - I_true))
    total_loss = lambda_1 * voltage_mse_norm + lambda_2 * current_mae_norm
    return total_loss, voltage_mse_norm, current_mae_norm


# Training
if __name__ == "__main__":
    torch.manual_seed(42)
    directory = "data/33bus2"
    dataset, Ybus = load_33bus_data(directory)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    input_dim = dataset[0].x.shape[1]
    model = GNNPINN(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    lambda_1, lambda_2 = 1.0, 0.0
    adjustment_step = 0.05
    adjustment_epochs = 500
    epochs = 10000

    epoch_log, voltage_loss_log, current_loss_log = [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss_epoch, volt_loss_epoch, curr_loss_epoch = 0, 0, 0
        for batch in loader:
            pred = model(batch)
            V_mag_pred, V_ang_pred = pred[:, 0], pred[:, 1]
            V_pred = V_mag_pred * torch.exp(1j * V_ang_pred)

            loss, v_loss, c_loss = pinn_loss(
                V_pred, batch.V_true, batch.I_true, Ybus, lambda_1, lambda_2
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            volt_loss_epoch += v_loss.item()
            curr_loss_epoch += c_loss.item()

        if (epoch + 1) % adjustment_epochs == 0:
            lambda_1 = max(0.0, lambda_1 - adjustment_step)
            lambda_2 = min(1.0, lambda_2 + adjustment_step)

        if epoch % 500 == 0 or epoch == epochs - 1:
            avg_v_loss = volt_loss_epoch / len(loader)
            avg_c_loss = curr_loss_epoch / len(loader)
            print(
                f"Epoch {epoch}: Total Loss={total_loss_epoch:.6f}, Voltage Loss={avg_v_loss:.6f}, Current Loss={avg_c_loss:.6f}, λ1={lambda_1:.2f}, λ2={lambda_2:.2f}"
            )
            epoch_log.append(epoch)
            voltage_loss_log.append(avg_v_loss)
            current_loss_log.append(avg_c_loss)

    # Save model
    torch.save(model.state_dict(), "gnn_pinn_model.pth")
    print("Model saved as gnn_pinn_model.pth")

    # Plot errors
    plt.figure(figsize=(10, 5))
    plt.plot(
        epoch_log, voltage_loss_log, label="Normalized Voltage MSE (u)", linewidth=2
    )
    plt.plot(
        epoch_log, current_loss_log, label="Normalized Current MAE (f)", linewidth=2
    )
    plt.xlabel("Epochs")
    plt.ylabel("Normalized Error")
    plt.title("GNN PINN - Voltage and Current Errors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
