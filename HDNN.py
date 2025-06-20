
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from ase.build import molecule
from ase import Atoms

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

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

# Improved training function with AdamW, weight decay, and a learning rate scheduler.
def train_model(model, train_loader, test_loader, device, epochs=300, lr=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    criterion = nn.L1Loss()  # Mean Absolute Error
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            x, y, q = batch  # x: representations, y: energies, q: charges
            x, y, q = x.to(device), y.to(device), q.to(device)
            optimizer.zero_grad()
            pred = model(x, q)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n_train += x.size(0)
        avg_train_loss = train_loss / n_train
        
        # Evaluate on the test set
        model.eval()
        total_loss = 0.0
        n_test = 0
        with torch.no_grad():
            for batch in test_loader:
                x, y, q = batch
                x, y, q = x.to(device), y.to(device), q.to(device)
                pred = model(x, q)
                total_loss += criterion(pred, y).item() * x.size(0)
                n_test += x.size(0)
        avg_test_loss = total_loss / n_test
        
        scheduler.step(avg_train_loss)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f}")
    
    return avg_test_loss

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

