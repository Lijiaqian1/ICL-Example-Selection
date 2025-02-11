import json
from penman import decode, Graph

input_file = "all_amrs.json"

decoded_graphs = []

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)  

    for entry in data:
        graph_str = entry.get("graph", "")
        if graph_str:
            try:
                
                graph = decode(graph_str)
                decoded_graphs.append((entry.get("sentence", ""), graph)) 
            except Exception as e:
                print(f"Failed to decode graph for ID {entry.get('id')}: {e}")

print(f"Successfully decoded {len(decoded_graphs)} AMR graphs.")

# demo
if len(decoded_graphs) > 9:
    sentence, graph = decoded_graphs[5]
    print(f"Sentence 10:\n{sentence}")
    print(f"Decoded Graph 10:\n{graph}")
