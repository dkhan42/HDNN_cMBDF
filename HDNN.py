import numpy as np
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

def pad_charges(q, pad):
    temp  = np.zeros(pad)
    size = len(q)
    temp[:size] = q
    return temp

# Define the atomic network (a simple feedforward NN)
class AtomicNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AtomicNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

# Define an element-specific HDNN.
class ElementSpecificNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, species):
        super(ElementSpecificNN, self).__init__()
        # Create a separate network for each element, using string keys.
        self.atomic_nns = nn.ModuleDict({
            str(int(s)): AtomicNN(input_dim, hidden_dim, output_dim) for s in species
        })
    
    def forward(self, x, charges):
        # x: (batch_size, num_atoms, input_dim)
        # charges: (batch_size, num_atoms), containing atomic numbers.
        batch_size, num_atoms, input_dim = x.size()
        x_flat = x.view(-1, input_dim)              # (batch_size*num_atoms, input_dim)
        charges_flat = charges.view(-1).float()       # (batch_size*num_atoms)
        atomic_energies = torch.zeros(x_flat.shape[0], device=x.device)
        
        # For each element, select the corresponding atoms and process with its network.
        for elem in self.atomic_nns.keys():
            mask = (charges_flat == float(elem))
            if mask.sum() > 0:
                x_elem = x_flat[mask]
                energy_elem = self.atomic_nns[elem](x_elem).squeeze()
                atomic_energies[mask] = energy_elem
        
        # Reshape to (batch_size, num_atoms) and sum atomic energies for each molecule.
        atomic_energies = atomic_energies.view(batch_size, num_atoms)
        total_energy = atomic_energies.sum(dim=1)
        return total_energy
    

def prepare_data_loaders(xtrain, ytrain, qtrain, batch_size=64, train_val_split=0.8):
    # Convert to tensors
    x_tensor = torch.tensor(xtrain, dtype=torch.float32)
    y_tensor = torch.tensor(ytrain, dtype=torch.float32)
    q_tensor = torch.tensor(qtrain, dtype=torch.float32)

    # Combine into one dataset
    full_dataset = TensorDataset(x_tensor, y_tensor, q_tensor)

    # Compute split sizes
    total_size = len(full_dataset)
    train_size = int(train_val_split * total_size)
    val_size = total_size - train_size

    # Split into training and validation sets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# Function for training the model
def train_model(
    model,
    xtrain,
    ytrain,
    qtrain,
    batch_size,
    device,
    train_val_split=0.8,
    epochs=300,
    lr=1e-3,                   #Initial learning rate
    weight_decay=1e-4,         #L2 reguralization
    es_patience=150,            # early stopping patience
    grad_clip=5.0,             # max norm for gradient clipping
    checkpoint_path="best_model.pth",
    lr_patience=30             # Patience for reducing learning rate
):
    # prepare data
    train_loader, val_loader = prepare_data_loaders(
        xtrain, ytrain, qtrain, batch_size, train_val_split
    )

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=lr_patience, factor=0.5
    )
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    es_counter = 0

    train_losses, val_losses = [], []

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        # ——— Training ———
        model.train()
        running_loss = 0.0
        n_train = 0
        for x, y, q in train_loader:
            x, y, q = x.to(device), y.to(device), q.to(device)
            optimizer.zero_grad()
            preds = model(x, q)
            loss = criterion(preds, y)
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n_train += x.size(0)

        avg_train = running_loss / n_train

        # ——— Validation ———
        model.eval()
        val_running = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y, q in val_loader:
                x, y, q = x.to(device), y.to(device), q.to(device)
                preds = model(x, q)
                val_running += criterion(preds, y).item() * x.size(0)
                n_val += x.size(0)

        avg_val = val_running / n_val

        # scheduler steps on validation, or you can change to avg_train
        scheduler.step(avg_val)

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        # print progress
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  •  train: {avg_train:.4e}  •  val: {avg_val:.4e}")

        # ——— Early Stopping & Checkpointing ———
        if avg_val < best_val:
            best_val = avg_val
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), checkpoint_path)
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= es_patience:
                print(f"Early stopping at epoch {epoch}, best val loss {best_val:.4e}")
                break

    # Restore best weights
    model.load_state_dict(best_model_wts)

    return model, np.array(train_losses), np.array(val_losses)

#Function for obtaining predictions using a trained model
def get_predictions(model, xtest, ytest, qtest, batch_size, device):
    x_test_tensor = torch.tensor(xtest, dtype=torch.float32)
    y_test_tensor = torch.tensor(ytest, dtype=torch.float32)
    q_test_tensor = torch.tensor(qtest, dtype=torch.float32)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor, q_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)

    preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc = 'Predicting batches: '):
            x, y, q = batch
            x, y, q = x.to(device), y.to(device), q.to(device)
            pred = model(x, q)
            preds.extend(pred)

    return np.array([x.cpu() for x in preds])
