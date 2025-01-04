import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

##################################################################
# 1) Contrastive Dataset
##################################################################

class ContrastivePairsDataset(Dataset):
    """
    Loads anchor, candidate, label from .npz => anchor (N, hidden_dim),
    candidate (N, hidden_dim), label = "pos"/"neg".
    We'll do a margin-based contrastive approach.
    """
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.anchor = data["anchor"]         # shape (N, hidden_dim)
        self.candidate = data["candidate"]   # shape (N, hidden_dim)
        self.labels = data["label"]          # shape (N,) => "pos"/"neg"

        # Convert to float32, in case it's something else
        # Also check if any are nan/inf; clamp them
        self.anchor = self._sanitize(self.anchor)
        self.candidate = self._sanitize(self.candidate)

        # Convert pos->1, neg->0
        self.bin_labels = np.array([1 if lab=="pos" else 0 
                                    for lab in self.labels], dtype=np.float32)

    def _sanitize(self, arr):
        # A small function that replaces potential nan or inf with finite values
        arr = arr.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        # if needed, we can also clamp the magnitude
        np.clip(arr, -1e6, 1e6, out=arr)
        return arr

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        a = self.anchor[idx]
        c = self.candidate[idx]
        label = self.bin_labels[idx]
        return torch.tensor(a), torch.tensor(c), torch.tensor(label)


##################################################################
# 2) SiameseProbe
##################################################################

class SiameseProbe(nn.Module):
    """
    A small MLP that maps each hidden vector to an embedding space.
    We'll do margin-based contrastive loss on the mapped embeddings.
    """
    def __init__(self, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        # x shape => (batch, hidden_dim)
        return self.net(x)  # => (batch, out_dim)


##################################################################
# 3) Contrastive Loss (Margin-based) - stable version
##################################################################

def contrastive_loss_fn(anchor_z, cand_z, labels, margin=1.0, eps=1e-9):
    """
    anchor_z, cand_z => (batch, out_dim)
    labels => (batch,) in {0,1}
    We'll do squared distance => d2 = sum((a_z - c_z)^2)
    Then dist = sqrt(d2 + eps).  (Add eps to avoid sqrt(0) => 0)
    
    - if label=1 => want dist -> 0 => loss_pos = d2
    - if label=0 => want dist >= margin => loss_neg = max(0, margin - dist)^2
    """
    # shape => (batch, out_dim)
    diff = anchor_z - cand_z
    d2   = (diff*diff).sum(dim=1)  # shape => (batch,)
    # Add eps and sqrt
    dist = torch.sqrt(d2 + eps)    # shape => (batch,)

    pos_mask = (labels == 1)
    neg_mask = (labels == 0)

    pos_dist = dist[pos_mask]
    neg_dist = dist[neg_mask]

    # If no positives or no negatives in batch, handle gracefully
    loss_pos = (pos_dist**2).mean() if pos_dist.numel()>0 else 0.0

    zero = torch.zeros_like(neg_dist)
    loss_neg = (torch.relu(margin - neg_dist)**2).mean() if neg_dist.numel()>0 else 0.0

    loss = loss_pos + loss_neg
    return loss


##################################################################
# 4) Training loop
##################################################################

def train_siamese_probe(
    dataset,
    siamese_probe,
    epochs=10,
    batch_size=64,
    lr=1e-4,
    val_ratio=0.1,
    margin=1.0,
    grad_clip=5.0
):
    """
    - dataset: ContrastivePairsDataset
    - siamese_probe: the MLP mapping
    - lr=1e-4 (lower than before to reduce the chance of exploding gradients)
    - margin=1.0 => you can tune
    - grad_clip => clip grad norm to avoid overflow
    """
    device = next(siamese_probe.parameters()).device
    val_size = int(len(dataset)*val_ratio)
    train_size= len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = optim.Adam(siamese_probe.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        # train
        siamese_probe.train()
        total_loss=0.0

        for anchor_t, cand_t, label_t in train_loader:
            anchor_t = anchor_t.to(device).float()
            cand_t   = cand_t.to(device).float()
            label_t  = label_t.to(device)

            optimizer.zero_grad()
            a_z = siamese_probe(anchor_t)
            c_z = siamese_probe(cand_t)

            loss = contrastive_loss_fn(a_z, c_z, label_t, margin=margin)
            if torch.isnan(loss):
                print("Warning: encountered NaN loss in training batch, skipping.")
                continue

            loss.backward()

            # gradient clip to avoid explosion
            torch.nn.utils.clip_grad_norm_(siamese_probe.parameters(), grad_clip)

            optimizer.step()
            total_loss += loss.item() * label_t.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # validation
        siamese_probe.eval()
        val_loss=0.0
        with torch.no_grad():
            for anchor_t, cand_t, label_t in val_loader:
                anchor_t = anchor_t.to(device).float()
                cand_t   = cand_t.to(device).float()
                label_t  = label_t.to(device)

                a_z = siamese_probe(anchor_t)
                c_z = siamese_probe(cand_t)
                loss_b = contrastive_loss_fn(a_z, c_z, label_t, margin=margin)
                val_loss += loss_b.item()* label_t.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch}/{epochs} => train_loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}")

    print("Training finished.")


##################################################################
# 5) Main
##################################################################

def main():
    npz_path = "llama2_reps.npz"  # where anchor, candidate, label stored
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) load dataset
    ds = ContrastivePairsDataset(npz_path)
    print(f"Dataset size: {len(ds)}. anchor shape: {ds.anchor.shape}")
    hidden_dim = ds.anchor.shape[1]  # e.g. 4096

    # 2) create siamese probe
    out_dim = 256
    net = SiameseProbe(hidden_dim=hidden_dim, out_dim=out_dim)
    net.to(device)

    # 3) train
    train_siamese_probe(
        dataset=ds,
        siamese_probe=net,
        epochs=10,
        batch_size=64,
        lr=1e-4,         # lowered from 1e-3
        val_ratio=0.1,
        margin=1.0,
        grad_clip=5.0
    )

    # 4) save
    torch.save(net.state_dict(), "siamese_probe.pth")
    print("Saved siamese_probe.pth")


if __name__=="__main__":
    main()
