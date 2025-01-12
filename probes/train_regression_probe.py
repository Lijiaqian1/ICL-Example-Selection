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
    本次任务直接使用parse_similarity作为回归目标。
    """
    def __init__(self, npz_path, use_parse_sim=False):
        data = np.load(npz_path, allow_pickle=True)
        # data.keys => ['anchor', 'candidate', 'label', 'parse_similarity']
        self.anchor = data["anchor"]         # shape (N, hidden_dim)
        self.candidate = data["candidate"]   # shape (N, hidden_dim)
        self.parse_sim = data["parse_similarity"]  # shape (N,)
        self.use_parse_sim = use_parse_sim

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        anchor_vec = self.anchor[idx]     # shape (hidden_dim,)
        cand_vec   = self.candidate[idx]  # shape (hidden_dim,)
        parse_sim  = self.parse_sim[idx]  # float

        # 转成 torch tensor
        anchor_t = torch.tensor(anchor_vec, dtype=torch.float32)
        cand_t   = torch.tensor(cand_vec, dtype=torch.float32)
        psim_t   = torch.tensor(parse_sim, dtype=torch.float32)

        # 将 parse_sim 作为回归的目标输出
        return anchor_t, cand_t, psim_t

########################################################
# 2. 定义 MLP probe
########################################################

class MLPProbe(nn.Module):
    """
    一个简单的 MLP，用于对 (anchor, candidate) 预测 parse_similarity（回归任务）。
    """
    def __init__(self, hidden_dim, out_dim=1):
        super().__init__()
        # 这里使用相同的MLP结构: concat => 2 * hidden_dim => 256 => 128 => 64 => 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  
            nn.ReLU(),
            nn.Linear(256, 128),            
            nn.ReLU(),
            nn.Linear(128, 64),             
            nn.ReLU(),
            nn.Linear(64, out_dim)   
        )
        # 输出单个连续值

    def forward(self, anchor_vec, cand_vec):
        # anchor_vec shape: (batch, hidden_dim)
        # cand_vec shape:   (batch, hidden_dim)
        x = torch.cat([anchor_vec, cand_vec], dim=1)  # shape => (batch, 2*hidden_dim)
        output = self.fc(x)  # shape => (batch, 1)
        return output.squeeze(dim=1)  # squeeze成 (batch,)

########################################################
# 3. 训练函数（回归任务）加入早停机制
########################################################

def train_probe(
    dataset,
    model,
    max_epochs=100,
    batch_size=64,
    lr=1e-3,
    val_ratio=0.1,
    patience=10  # 连续不改进epoch数
):
    """
    dataset: ContrastivePairsDataset
    model: MLPProbe
    """
    # 拆分训练/验证集
    val_size = int(len(dataset)*val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = next(model.parameters()).device  # assume model on device
    print(f"Start training: max_epochs={max_epochs}, device={device}, train_size={train_size}, val_size={val_size}")

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, max_epochs+1):
        model.train()
        total_loss = 0.0
        for anchor_t, cand_t, target_sim in train_loader:
            anchor_t = anchor_t.to(device)
            cand_t   = cand_t.to(device)
            target_sim = target_sim.to(device)

            optimizer.zero_grad()
            outputs = model(anchor_t, cand_t)  # shape (batch,)
            loss = criterion(outputs, target_sim)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * anchor_t.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor_t, cand_t, target_sim in val_loader:
                anchor_t = anchor_t.to(device)
                cand_t   = cand_t.to(device)
                target_sim = target_sim.to(device)

                outputs = model(anchor_t, cand_t)
                loss = criterion(outputs, target_sim)
                val_loss += loss.item() * anchor_t.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch}/{max_epochs} => train_loss={avg_loss:.6f}, val_loss={avg_val_loss:.6f}")

        # 早停逻辑
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # 保存当前最优模型
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early Stop Triggered")
            break

    print("Training finished.")
    # 加载最佳模型参数（可选）
    model.load_state_dict(torch.load("best_model.pth"))
    torch.save(model.state_dict(), "mlp_probe_regression.pth")
    print("Model saved to mlp_probe_regression.pth.")

########################################################
# 4. 主流程
########################################################

def main():
    npz_path = "llama2_reps.npz"  # 数据文件路径
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 dataset
    ds = ContrastivePairsDataset(npz_path, use_parse_sim=True)
    print(f"Dataset size: {len(ds)}")

    # 构建 MLP 模型
    hidden_dim = ds.anchor.shape[1]  # shape => (N, hidden_dim)
    model = MLPProbe(hidden_dim=hidden_dim)
    model.to(device)

    # 训练回归模型，最多100个epoch，早停patience设为5
    train_probe(
        dataset=ds,
        model=model,
        max_epochs=100,
        batch_size=64,
        lr=1e-3,
        val_ratio=0.1,
        patience=5
    )

if __name__=="__main__":
    main()
