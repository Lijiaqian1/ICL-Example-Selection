import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

class SiameseProbe(nn.Module):
    def __init__(self, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class ContrastivePairsWithSimDataset(Dataset):
    def __init__(self, npz_path, json_path):
        data = np.load(npz_path, allow_pickle=True)
        self.anchor = data["anchor"]
        self.candidate = data["candidate"]
        self.labels = data["label"]
        with open(json_path, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        self.similarity = np.array([pair["parse_similarity"] for pair in pairs], dtype=np.float32)
        self.anchor = self._sanitize(self.anchor)
        self.candidate = self._sanitize(self.candidate)
        self.bin_labels = np.array([1 if lab == "pos" else 0 for lab in self.labels], dtype=np.float32)

    def _sanitize(self, arr):
        arr = arr.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        np.clip(arr, -1e6, 1e6, out=arr)
        return arr

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        a = torch.tensor(self.anchor[idx])
        c = torch.tensor(self.candidate[idx])
        label = torch.tensor(self.bin_labels[idx])
        sim = torch.tensor(self.similarity[idx])
        return a, c, label, sim

def contrastive_loss_with_similarity(anchor_z, cand_z, labels, similarities, margin=1.0, eps=1e-9):
    diff = anchor_z - cand_z
    d2 = (diff * diff).sum(dim=1)
    dist = torch.sqrt(d2 + eps)

    pos_mask = (labels == 1)
    neg_mask = (labels == 0)

    pos_sim = similarities[pos_mask]
    pos_targets = 1.0 - pos_sim
    pos_dist = dist[pos_mask]
    loss_pos = ((pos_dist - pos_targets) ** 2).mean() if pos_dist.numel() > 0 else 0.0

    neg_dist = dist[neg_mask]
    loss_neg = (torch.relu(margin - neg_dist) ** 2).mean() if neg_dist.numel() > 0 else 0.0

    return loss_pos + loss_neg

def train_probe_with_similarity(
    dataset,
    siamese_probe,
    epochs=10,
    batch_size=64,
    lr=1e-4,
    val_ratio=0.1,
    margin=1.0,
    grad_clip=5.0,
    patience=5
):
    device = next(siamese_probe.parameters()).device
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = optim.Adam(siamese_probe.parameters(), lr=lr)

    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        siamese_probe.train()
        total_loss = 0.0

        for anchor_t, cand_t, label_t, sim_t in train_loader:
            anchor_t = anchor_t.to(device).float()
            cand_t = cand_t.to(device).float()
            label_t = label_t.to(device)
            sim_t = sim_t.to(device)

            optimizer.zero_grad()
            a_z = siamese_probe(anchor_t)
            c_z = siamese_probe(cand_t)

            loss = contrastive_loss_with_similarity(a_z, c_z, label_t, sim_t, margin=margin)
            if torch.isnan(loss):
                print("Warning: NaN loss, skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(siamese_probe.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item() * label_t.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        siamese_probe.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor_t, cand_t, label_t, sim_t in val_loader:
                anchor_t = anchor_t.to(device).float()
                cand_t = cand_t.to(device).float()
                label_t = label_t.to(device)
                sim_t = sim_t.to(device)

                a_z = siamese_probe(anchor_t)
                c_z = siamese_probe(cand_t)
                loss_b = contrastive_loss_with_similarity(a_z, c_z, label_t, sim_t, margin=margin)
                val_loss += loss_b.item() * label_t.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch}/{epochs} => train_loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(siamese_probe.state_dict(), "best_siamese_probe_with_sim.pth")
            #print("Validation loss improved. Model saved.")
        else:
            no_improve_epochs += 1
            #print(f"No improvement for {no_improve_epochs} epochs.")

        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break

def main():
    npz_path = "../../probes/llama2_smcalflow_reps.npz"
    json_path = "smcalflow_contrastive_pairs.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ContrastivePairsWithSimDataset(npz_path, json_path)
    print(f"Dataset size: {len(dataset)}. Hidden dim: {dataset.anchor.shape[1]}")

    hidden_dim = dataset.anchor.shape[1]
    out_dim = 256

    siamese_probe = SiameseProbe(hidden_dim=hidden_dim, out_dim=out_dim)
    siamese_probe.to(device)

    train_probe_with_similarity(
        dataset=dataset,
        siamese_probe=siamese_probe,
        epochs=100,
        batch_size=64,
        lr=1e-4,
        val_ratio=0.1,
        margin=1.0,
        grad_clip=5.0,
        patience=30
    )

if __name__ == "__main__":
    main()
