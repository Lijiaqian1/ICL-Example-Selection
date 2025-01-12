import re
import numpy as np
from zss import Node, distance
from sentence_transformers import SentenceTransformer


embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')



def parse_lispress(lispress_str):
    tokens = tokenize_lispress(lispress_str)
    root, _ = build_tree(tokens, 0)
    return root

def tokenize_lispress(lispress_str):
    s = lispress_str.replace("(", " ( ").replace(")", " ) ")
    raw_tokens = s.split()
    tokens = [t for t in raw_tokens if t.strip()]
    return tokens

def build_tree(tokens, start_idx):
    idx = start_idx
    if tokens[idx] != "(":
        leaf_node = Node(tokens[idx])
        return leaf_node, idx + 1

    idx += 1
    label = tokens[idx]
    root = Node(label)
    idx += 1

    while idx < len(tokens) and tokens[idx] != ")":
        if tokens[idx] == "(":
            child_tree, new_pos = build_tree(tokens, idx)
            root.addkid(child_tree)
            idx = new_pos
        else:
            leaf_node = Node(tokens[idx])
            root.addkid(leaf_node)
            idx += 1

    idx += 1  # consume ")"
    return root, idx



def is_simple_identifier(label):
    return not ("." in label or "(" in label or ")" in label)

def label_distance(labelA, labelB):
    if labelA == labelB:
        return 0  
    if is_simple_identifier(labelA) and is_simple_identifier(labelB):
        return 0.4
    elif is_simple_identifier(labelA) or is_simple_identifier(labelB):
        return 0.6
    if not is_simple_identifier(labelA) and not is_simple_identifier(labelB):
        return embedding_distance(labelA, labelB) * 0.1


    return 1.0

def embedding_distance(text1, text2):
    embeddings = embedder.encode([text1, text2])
    vec1, vec2 = embeddings[0], embeddings[1]
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    dist = 1 - cos_sim  
    return dist

def node_distance(nodeA: Node, nodeB: Node):
    if nodeA is None or nodeB is None:
        return 1  
    return label_distance(nodeA.label, nodeB.label)


def lispress_tree_distance(parseA, parseB):
    treeA = parse_lispress(parseA)
    treeB = parse_lispress(parseB)

    dist = distance(
        treeA,
        treeB,
        get_children=lambda n: n.children,
        insert_cost=lambda n: 1,  
        remove_cost=lambda n: 1,  
        update_cost=node_distance  
    )
    return dist

def lispress_tree_similarity(parseA, parseB):

    dist = lispress_tree_distance(parseA, parseB)
    return 1 / (1 + dist)


if __name__ == "__main__":
    p1 = "(Yield (Tomorrow))"
    p2 = "(Yield (Date.dayOfWeek (Tomorrow)))"
    p3 = "(Yield (Date.dayOfWeek (Date.yesterday)))"

    d12 = lispress_tree_distance(p1, p2)
    s12 = lispress_tree_similarity(p1, p2)
    print(f"p1 vs p2 => dist={d12}, sim={s12:.3f}")

    d13 = lispress_tree_distance(p1, p3)
    s13 = lispress_tree_similarity(p1, p3)
    print(f"p1 vs p3 => dist={d13}, sim={s13:.3f}")

    d23 = lispress_tree_distance(p2, p3)
    s23 = lispress_tree_similarity(p2, p3)
    print(f"p2 vs p3 => dist={d23}, sim={s23:.3f}")
