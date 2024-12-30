import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


with open("../data/amrs/split/training/all_amrs.json", "r", encoding="utf-8") as f:
    all_amrs = json.load(f)
    
# all_amrs 现在是一个列表，每个元素形如:
# { "id": str, "sentence": str, "graph": ... }

print("Total AMRs:", len(all_amrs))

# 2. 准备句子列表
sentences = [item["sentence"] for item in all_amrs]

# 3. 初始化模型
model = SentenceTransformer("all-MiniLM-L6-v2")  
# 你也可以选其他模型，越大效果越好，但也更慢

# 4. 生成句子向量
embeddings = model.encode(
    sentences, 
    batch_size=32, 
    convert_to_numpy=True, 
    show_progress_bar=True
)  # shape: (N, hidden_dim)

# 转成 float32 以适配 faiss
embeddings = embeddings.astype(np.float32)
# 5. 构建索引 (使用内积IndexFlatIP最简单)
d = embeddings.shape[1]  # 向量维度
index = faiss.IndexFlatIP(d)  # IP = inner product
index.add(embeddings)         # 将所有向量加入索引

# 6. 对所有句子做检索，取 topK
K = 10
D, I = index.search(embeddings, K)  
# D.shape = (N, K), I.shape = (N, K)
# 对于每个 i, I[i] 是最相似的K个条目下标, D[i]是对应余弦(内积)相似度

print("Index search completed.")

np.save("I.npy", I)
np.save("D.npy", D)

print("I.npy and D.npy have been saved successfully.")