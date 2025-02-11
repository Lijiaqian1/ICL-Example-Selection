import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# -------------------------
# 1) Dataset
# -------------------------
class ContrastiveInfoNCEDataset(Dataset):
    """
    Loads an NPZ file containing:
      - anchor: shape (N, hidden_dim)
      - positive: shape (N, hidden_dim)
      - negatives: object array of shape (N,) where each entry is (n_neg, hidden_dim)
    
    Returns (anchor, positive, negatives) for each item.
    """
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.anchor   = data["anchor"]     
        self.positive = data["positive"]    
        self.negatives = data["negatives"]  
        self.anchor   = self._sanitize(self.anchor)
        self.positive = self._sanitize(self.positive)
        
   
        self.negatives_list = []
        for neg_array in self.negatives:
            neg_array = self._sanitize(neg_array)
            self.negatives_list.append(neg_array)
    
    def _sanitize(self, arr):
        arr = arr.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        np.clip(arr, -1e6, 1e6, out=arr)
        return arr

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        a = self.anchor[idx]
        p = self.positive[idx]
        n = self.negatives_list[idx]  # shape: (n_neg, hidden_dim)
        # Return anchor, positive, negatives
        return torch.tensor(a), torch.tensor(p), torch.tensor(n)


# -------------------------
# 2) Siamese Network (MLP)
# -------------------------
class SiameseProbe(nn.Module):
    """
    A simple feed-forward MLP that projects hidden_dim -> out_dim.
    """
    def __init__(self, hidden_dim=4096, out_dim=256):
        super(SiameseProbe, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(512, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)  # shape: (batch, out_dim)


# -------------------------
# 3) InfoNCE Loss
# -------------------------
def info_nce_loss(anchor_z, positive_z, negatives_z, temperature=0.07):
    """
    anchor_z:   (B, d)
    positive_z: (B, d)
    negatives_z: shape (B, n_neg, d)
    
    We'll compute the similarity anchor<->positive and anchor<->negatives,
    then apply the InfoNCE objective:
    
      L = -log( exp(sim(a,p)/T) / (exp(sim(a,p)/T) + sum_{neg} exp(sim(a,neg)/T)) )
    
    where sim is the cosine similarity.
    """
    # Normalize
    anchor_norm   = nn.functional.normalize(anchor_z, p=2, dim=1)
    positive_norm = nn.functional.normalize(positive_z, p=2, dim=1)
    # negatives_z: (B, n_neg, d) => normalize along last dim
    negatives_norm = nn.functional.normalize(negatives_z, p=2, dim=2)

    # Compute cos sim for anchor vs positive => (B,)
    pos_sim = (anchor_norm * positive_norm).sum(dim=1)  # dot product
    # Compute cos sim for anchor vs each negative => (B, n_neg)
    # We can use bmm or broadcasting
    # anchor_norm shape: (B, d) => (B, 1, d)
    # negatives_norm shape: (B, n_neg, d)
    # => do (B, n_neg, d) * (B, 1, d) => still (B, n_neg, d)
    # sum over dim=2 => (B, n_neg)
    neg_sim = (negatives_norm * anchor_norm.unsqueeze(1)).sum(dim=2)

    # Combine [pos_sim, neg_sim], shape => (B, 1 + n_neg)
    # pos_sim => (B,)
    # neg_sim => (B, n_neg)
    # => cat along dim=1
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1 + n_neg)

    # scale by temperature
    logits = logits / temperature

    # The label for each row is 0 (the first column is the positive)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    # standard cross-entropy
    loss = nn.functional.cross_entropy(logits, labels)
    return loss


# -------------------------
# 4) Training Function
# -------------------------
def train_infonce_probe(
    dataset,
    model,
    epochs=10,
    batch_size=64,
    lr=1e-4,
    val_ratio=0.1,
    temperature=0.05
):
    device = next(model.parameters()).device
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")  # 记录最优验证损失
    best_model_path = "siamese_probe_best.pth"  # 最优模型存储路径

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        total_loss = 0.0
        for anchor_t, positive_t, negatives_t in tqdm(train_loader, desc=f"Epoch {epoch} (train)"):
            anchor_t, positive_t, negatives_t = anchor_t.to(device), positive_t.to(device), negatives_t.to(device)
            optimizer.zero_grad()

            a_z, p_z = model(anchor_t), model(positive_t)
            B, n_neg, hid_dim = negatives_t.shape
            negatives_z = model(negatives_t.view(-1, hid_dim)).view(B, n_neg, -1)

            loss = info_nce_loss(a_z, p_z, negatives_z, temperature=temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * anchor_t.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor_t, positive_t, negatives_t in val_loader:
                anchor_t, positive_t, negatives_t = anchor_t.to(device), positive_t.to(device), negatives_t.to(device)
                
                a_z, p_z = model(anchor_t), model(positive_t)
                B, n_neg, hid_dim = negatives_t.shape
                negatives_z = model(negatives_t.view(-1, hid_dim)).view(B, n_neg, -1)
                
                loss_b = info_nce_loss(a_z, p_z, negatives_z, temperature=temperature)
                val_loss += loss_b.item() * anchor_t.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch}/{epochs} => train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # ---- Save Best Model ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val_loss={best_val_loss:.4f}")

    print("Training finished. Best model saved at:", best_model_path)



# -------------------------
# 5) Main
# -------------------------
def main():
    #npz_path = "llama2_reps_infonce.npz"  # the new NPZ file you created
    npz_path = "llama2_reps_infonce_with_embedding_filter.npz"
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    # Build dataset
    ds = ContrastiveInfoNCEDataset(npz_path)
    print(f"Dataset size: {len(ds)}")

    # We assume anchor shape => (N, hidden_dim)
    anchor_dim = ds.anchor.shape[1]
    out_dim = 256

    # Build model
    net = SiameseProbe(hidden_dim=anchor_dim, out_dim=out_dim)
    net.to(device)

    # Train
    train_infonce_probe(
        dataset=ds,
        model=net,
        epochs=10,
        batch_size=64,
        lr=1e-4,
        val_ratio=0.1,
        temperature=0.1
    )

    # Save model
    #torch.save(net.state_dict(), "siamese_probe_infonce.pth")
    torch.save(net.state_dict(), "siamese_probe_infonce_with_embedding_filter.pth")
    print("Saved siamese_probe_infonce.pth")


if __name__ == "__main__":
    main()
