import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

########################################################
# 1. 准备数据集
########################################################

class ContrastivePairsDataset(Dataset):
    """
    一个简单的Dataset，用于从 .npz 文件加载:
      anchor, candidate, label, parse_similarity
    并将 pos/neg label 转为 [1,0], parse_similarity可选。
    """
    def __init__(self, npz_path, use_parse_sim=False):
        """
        use_parse_sim = True 时，可以在 forward 里使用 parse_similarity 做某些混合损失。
        """
        data = np.load(npz_path, allow_pickle=True)
        # data.keys => ['anchor', 'candidate', 'label', 'parse_similarity']
        self.anchor = data["anchor"]   # shape (N, hidden_dim)
        self.candidate = data["candidate"] # shape (N, hidden_dim)
        self.labels = data["label"]    # shape (N,) => e.g. 'pos'/'neg'
        self.parse_sim = data["parse_similarity"]  # shape (N,)
        self.use_parse_sim = use_parse_sim

        # 将字符串标签 pos/neg 转化成 1/0
        self.bin_labels = np.array([1 if lab=="pos" else 0 for lab in self.labels], dtype=np.float32)

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        anchor_vec = self.anchor[idx]     # shape (hidden_dim,)
        cand_vec   = self.candidate[idx]  # shape (hidden_dim,)
        label_val  = self.bin_labels[idx] # 1 or 0
        parse_sim  = self.parse_sim[idx]  # float

        # 转成 torch
        anchor_t = torch.tensor(anchor_vec, dtype=torch.float32)
        cand_t   = torch.tensor(cand_vec, dtype=torch.float32)
        label_t  = torch.tensor(label_val, dtype=torch.float32)
        psim_t   = torch.tensor(parse_sim, dtype=torch.float32)

        return anchor_t, cand_t, label_t, psim_t


########################################################
# 2. 定义 MLP probe
########################################################

class MLPProbe(nn.Module):
    """
    一个简单的 MLP，用于对 (anchor, candidate) 做二分类。
    常见做法:
      - 先做 difference or concat => fc => ReLU => ... => output => sigmoid
    """
    def __init__(self, hidden_dim, out_dim=1):
        super().__init__()
        # 这里举例: concat => 2 * hidden_dim => 256 => 64 => 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  # 第一隐藏层
            nn.ReLU(),
            nn.Linear(256, 128),            # 第二隐藏层
            nn.ReLU(),
            nn.Linear(128, 64),             # 第三隐藏层
            nn.ReLU(),
            nn.Linear(64, out_dim)   
        )
        # 最后一层输出 raw logits => 用 BCEWithLogitsLoss
    def forward(self, anchor_vec, cand_vec):
        # anchor_vec shape: (batch, hidden_dim)
        # cand_vec shape:   (batch, hidden_dim)
        x = torch.cat([anchor_vec, cand_vec], dim=1)  # shape => (batch, 2*hidden_dim)
        logits = self.fc(x)  # shape => (batch, 1)
        return logits



def train_probe(
    dataset,
    model,
    epochs=5,
    batch_size=64,
    lr=1e-3,
    val_ratio=0.1,
    parse_sim_weight=False
):


    val_size = int(len(dataset)*val_ratio)
    train_size= len(dataset)- val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = next(model.parameters()).device  
    print(f"Start training: epochs={epochs}, device={device}, train_size={train_size}, val_size={val_size}")

    for epoch in range(1, epochs+1):
        model.train()
        total_loss=0
        for anchor_t, cand_t, label_t, psim_t in train_loader:
            anchor_t = anchor_t.to(device)
            cand_t   = cand_t.to(device)
            label_t  = label_t.to(device)

            optimizer.zero_grad()
            logits = model(anchor_t, cand_t)  # shape (batch,1)
            logits = logits.squeeze(dim=1)    # shape (batch,)
            loss   = criterion(logits, label_t)

            # if parse_sim_weight => extra term. e.g. L += alpha * MSE( logits.sigmoid(), psim_t??? ) 
            # but let's keep it simple

            loss.backward()
            optimizer.step()

            total_loss += loss.item()* len(label_t)

        avg_loss = total_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss=0
        correct=0
        total=0
        with torch.no_grad():
            for anchor_t, cand_t, label_t, psim_t in val_loader:
                anchor_t = anchor_t.to(device)
                cand_t   = cand_t.to(device)
                label_t  = label_t.to(device)

                logits = model(anchor_t, cand_t).squeeze(dim=1)
                loss   = criterion(logits, label_t)

                val_loss += loss.item()* len(label_t)
                # compute accuracy
                preds = torch.sigmoid(logits) > 0.5
                correct += (preds.float() == label_t).sum().item()
                total   += len(label_t)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        print(f"Epoch {epoch}/{epochs} => train_loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}")
    print("Training finished.")
    torch.save(model.state_dict(), "mlp_probe.pth")
    print("Model saved to mlp_probe.pth.")
    


########################################################
# 4. 主流程
########################################################

def main():
    npz_path = "llama2_reps.npz"  # 你前面保存的
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载dataset
    ds = ContrastivePairsDataset(npz_path, use_parse_sim=False)
    print(f"Dataset size: {len(ds)}")

    # 构建MLP
    hidden_dim= ds.anchor.shape[1]  # shape => (N, hidden_dim)
    model = MLPProbe(hidden_dim=hidden_dim)
    model.to(device)

    # 训练
    train_probe(
        dataset=ds,
        model=model,
        epochs=10,
        batch_size=64,
        lr=1e-3,
        val_ratio=0.1,
        parse_sim_weight=False
    )

if __name__=="__main__":
    main()
