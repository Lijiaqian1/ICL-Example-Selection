import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# -------------------------
# 1) Dataset
# -------------------------
class ContrastiveInfoNCEDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.anchor = data["anchor"]
        self.positive = data["positive"]
        self.negatives = data["negatives"]

        self.anchor = self._sanitize(self.anchor)
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
        n = self.negatives_list[idx]
        return torch.tensor(a), torch.tensor(p), torch.tensor(n)


# -------------------------
# 2) Dual Encoder with Interaction Module
# -------------------------
class DualEncoderWithInteraction(nn.Module):
    def __init__(self, hidden_dim=4096, out_dim=256, interaction_mode="concat"):
        super(DualEncoderWithInteraction, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        self.interaction_mode = interaction_mode

        if interaction_mode == "concat":
            self.interaction_layer = nn.Sequential(
                nn.Linear(2 * out_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)  # Output a scalar score
            )
        elif interaction_mode == "dot":
            self.interaction_layer = None  # Dot product has no learnable parameters
        elif interaction_mode == "attention":
            self.interaction_layer = nn.MultiheadAttention(embed_dim=out_dim, num_heads=4)
            self.final_fc = nn.Linear(out_dim, 1)  # Output a scalar score
        else:
            raise ValueError("interaction_mode must be one of ['concat', 'dot', 'attention']")

    def forward(self, anchor, candidate):
        a_z = self.encoder(anchor)
        c_z = self.encoder(candidate)

        if self.interaction_mode == "concat":
            combined = torch.cat([a_z, c_z], dim=-1)
            score = self.interaction_layer(combined).squeeze(-1)  # Shape: (B,)
        elif self.interaction_mode == "dot":
            score = (a_z * c_z).sum(dim=-1)  # Dot product similarity
        elif self.interaction_mode == "attention":
            a_z_exp = a_z.unsqueeze(0)  # (B, d) -> (1, B, d) for attention
            c_z_exp = c_z.unsqueeze(0)  # (B, d) -> (1, B, d)
            attn_output, _ = self.interaction_layer(a_z_exp, c_z_exp, c_z_exp)
            score = self.final_fc(attn_output.squeeze(0)).squeeze(-1)  # Shape: (B,)
        return score


# -------------------------
# 3) InfoNCE Loss
# -------------------------
def info_nce_loss(anchor, positive, negatives, model, temperature=0.07):
    """
    Computes the InfoNCE loss:
    
      L = -log( exp(sim(a,p)/T) / (exp(sim(a,p)/T) + sum_{neg} exp(sim(a,neg)/T)) )
    
    where sim is computed based on the selected interaction mode.
    """
    pos_sim = model(anchor, positive)  # (B,)
    neg_sim = torch.stack([model(anchor, neg) for neg in negatives.permute(1, 0, 2)], dim=1)  # (B, n_neg)

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1 + n_neg)
    logits = logits / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    loss = nn.functional.cross_entropy(logits, labels)
    return loss


# -------------------------
# 4) Training Function
# -------------------------
def train_dual_encoder(
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
    best_val_loss = float("inf")
    best_model_path = "dual_encoder_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for anchor, positive, negatives in tqdm(train_loader, desc=f"Epoch {epoch} (train)"):
            anchor, positive, negatives = anchor.to(device), positive.to(device), negatives.to(device)
            optimizer.zero_grad()
            loss = info_nce_loss(anchor, positive, negatives, model, temperature=temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * anchor.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negatives in val_loader:
                anchor, positive, negatives = anchor.to(device), positive.to(device), negatives.to(device)
                loss = info_nce_loss(anchor, positive, negatives, model, temperature=temperature)
                val_loss += loss.item() * anchor.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch}/{epochs} => train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val_loss={best_val_loss:.4f}")

    print("Training finished. Best model saved at:", best_model_path)


# -------------------------
# 5) Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction_mode", type=str, choices=["concat", "dot", "attention"], required=True)
    args = parser.parse_args()

    npz_path = "llama2_reps_infonce.npz"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ContrastiveInfoNCEDataset(npz_path)
    model = DualEncoderWithInteraction(hidden_dim=ds.anchor.shape[1], out_dim=256, interaction_mode=args.interaction_mode)
    model.to(device)

    train_dual_encoder(dataset=ds, model=model, epochs=50, batch_size=64, lr=1e-4, val_ratio=0.1, temperature=0.07)

    model_name = args.interaction_mode + "_dual_encoder.pth"

    torch.save(model.state_dict(), model_name)
    print("Saved dual_encoder.pth")


if __name__ == "__main__":
    main()
