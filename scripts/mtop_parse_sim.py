import re
import numpy as np
from zss import Node, distance
from sentence_transformers import SentenceTransformer


embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

def embedding_distance(text1, text2):

    embeddings = embedder.encode([text1, text2])
    vec1, vec2 = embeddings[0], embeddings[1]
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    dist = 1 - cos_sim  
    return dist



def parse_mtop_bracket(parse_str):

    tokens = tokenize_bracket(parse_str)
    root, _ = build_tree(tokens, 0)
    reorder_children(root) 
    return root

def tokenize_bracket(parse_str):
    s = parse_str.replace("[", " [ ").replace("]", " ] ")
    raw_tokens = s.split()
    tokens = [t for t in raw_tokens if t.strip()]
    return tokens

def build_tree(tokens, start_idx):
    idx = start_idx
    if tokens[idx] != "[":
        leaf_node = Node(tokens[idx])
        return leaf_node, idx+1

    idx += 1
    label = tokens[idx]
    root = Node(label)
    idx += 1

    while idx < len(tokens) and tokens[idx] != "]":
        if tokens[idx] == "[":
            child_tree, new_pos = build_tree(tokens, idx)
            root.addkid(child_tree)
            idx = new_pos
        else:
            leaf_node = Node(tokens[idx])
            root.addkid(leaf_node)
            idx += 1

    idx += 1  
    return root, idx

def reorder_children(root: Node):
    kids = root.children
    if not kids:
        return
    for c in kids:
        reorder_children(c)
    def parse_label(label):
        if label.startswith("IN:"):
            return ("IN", label[3:])
        elif label.startswith("SL:"):
            return ("SL", label[3:])
        else:
            return ("LEAF", label)
    kids.sort(key=lambda n: parse_label(n.label))



def label_distance(labelA, labelB):

    if labelA.startswith("IN:") and labelB.startswith("IN:"):
        intentA = labelA[3:]
        intentB = labelB[3:]
        
        prefixA = intentA.split("_")[0]
        prefixB = intentB.split("_")[0]
        
        if intentA == intentB:
            return 0  
        elif prefixA == prefixB:
            return 0.5  
        else:
            return 1  


    if labelA.startswith("SL:") and labelB.startswith("SL:"):
        slotA = labelA[3:]
        slotB = labelB[3:]
        return 0 if slotA==slotB else 0.5

    isLeafA = (not labelA.startswith("IN:")) and (not labelA.startswith("SL:"))
    isLeafB = (not labelB.startswith("IN:")) and (not labelB.startswith("SL:"))
    if isLeafA and isLeafB:
        return embedding_distance(labelA, labelB)*0.1

    return 1

def node_distance(nodeA: Node, nodeB: Node):
    if nodeA is None or nodeB is None:
        return 1 
    return label_distance(nodeA.label, nodeB.label)



def mtop_tree_distance(parseA, parseB):
    treeA = parse_mtop_bracket(parseA)
    treeB = parse_mtop_bracket(parseB)


    dist = distance(
        treeA,
        treeB,
        get_children=lambda n: n.children,
        insert_cost=lambda n: 1,  
        remove_cost=lambda n: 1,  
        update_cost=node_distance 
    )
    return dist

def mtop_tree_similarity(parseA, parseB):
    dist = mtop_tree_distance(parseA, parseB)
    return 1/(1+dist)


def embedding_similarity(sentence1, sentence2):
    embeddings = embedder.encode([sentence1, sentence2])
    vec1, vec2 = embeddings[0], embeddings[1]
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim

if __name__ == "__main__":
    p1 = "[IN:GET_MESSAGE [SL:TYPE_CONTENT video ] [SL:SENDER Atlas ] ]"
    p2 = "[IN:GET_MESSAGE [SL:SENDER Atlasss ] [SL:TYPE_CONTENT videos ] ]"
    p3 = "[IN:GET_MESSAGE [SL:TYPE_CONTENT video ] ]"

    p4 = "[IN:GET_WEATHER [SL:LOCATION Charlotte ] ]"
    p5 = "[IN:GET_EVENT [SL:DATE_TIME tonight ] ]"
    p6 = "[IN:GET_WEATHER [SL:LOCATION Maricopa County ] [SL:DATE_TIME for this week ] ]"

    s4 = "what is the weather like in Charlotte"
    s5 = "what is happening tonight"
    s6 = "Maricopa County weather forecast for this week"

    d12 = mtop_tree_distance(p4, p5)
    s12 = mtop_tree_similarity(p4, p5)
    print(f"p4 vs p5 => dist={d12}, sim={s12:.3f}")

    d13 = mtop_tree_distance(p4, p6)
    s13 = mtop_tree_similarity(p4, p6)
    print(f"p4 vs p6 => dist={d13}, sim={s13:.3f}")


    sim_s4_s5 = embedding_similarity(s4, s5)
    sim_s4_s6 = embedding_similarity(s4, s6)

    print(f"s4 vs s5 => similarity={sim_s4_s5:.3f}")
    print(f"s4 vs s6 => similarity={sim_s4_s6:.3f}")