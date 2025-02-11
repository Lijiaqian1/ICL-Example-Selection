import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


with open("../data/amrs/split/training/all_amrs.json", "r", encoding="utf-8") as f:
    all_amrs = json.load(f)
    


print("Total AMRs:", len(all_amrs))

sentences = [item["sentence"] for item in all_amrs]

model = SentenceTransformer("all-MiniLM-L6-v2")  

embeddings = model.encode(
    sentences, 
    batch_size=32, 
    convert_to_numpy=True, 
    show_progress_bar=True
)  
embeddings = embeddings.astype(np.float32)
d = embeddings.shape[1]  
index = faiss.IndexFlatIP(d)  
index.add(embeddings)         

K = 10
D, I = index.search(embeddings, K)  


print("Index search completed.")

np.save("I.npy", I)
np.save("D.npy", D)

print("I.npy and D.npy have been saved successfully.")