import re
import numpy as np
from zss import Node, distance
from sentence_transformers import SentenceTransformer

####################################
# 0. 全局准备
####################################

# 用于叶子embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

def embedding_distance(text1, text2):
    """用 Sentence-BERT 计算叶节点语义距离, 返回 [0,1+]"""
    embeddings = embedder.encode([text1, text2])
    vec1, vec2 = embeddings[0], embeddings[1]
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    dist = 1 - cos_sim  # 距离
    return dist

####################################
# 1. 解析 bracket -> zss.Node
#    + 无序化
####################################

def parse_mtop_bracket(parse_str):
    """
    将 MTop bracket parse => zss.Node 树.
    e.g. "[IN:GET_MESSAGE [SL:TYPE_CONTENT video ] [SL:SENDER Atlas ] ]"
    """
    tokens = tokenize_bracket(parse_str)
    root, _ = build_tree(tokens, 0)
    reorder_children(root)  # 无序 => sort children
    return root

def tokenize_bracket(parse_str):
    s = parse_str.replace("[", " [ ").replace("]", " ] ")
    raw_tokens = s.split()
    tokens = [t for t in raw_tokens if t.strip()]
    return tokens

def build_tree(tokens, start_idx):
    idx = start_idx
    if tokens[idx] != "[":
        # 叶子
        leaf_node = Node(tokens[idx])
        return leaf_node, idx+1

    # 否则是内节点
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
            # 叶子节点
            leaf_node = Node(tokens[idx])
            root.addkid(leaf_node)
            idx += 1

    idx += 1  # consume "]"
    return root, idx

def reorder_children(root: Node):
    """
    对每个节点的 children 按 label 排序, 以减少无序插入造成的编辑距离.
    """
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

####################################
# 2. 自定义 label_distance
####################################

def label_distance(labelA, labelB):
    """
    节点替换代价:
    1) 同为IN: => 0或1
    2) 同为SL: => 0或0.5
    3) 叶子 => embedding
    4) 不同type => 1
    """
    if labelA.startswith("IN:") and labelB.startswith("IN:"):
        intentA = labelA[3:]
        intentB = labelB[3:]
        
        # 获取第一个 "_" 前的部分
        prefixA = intentA.split("_")[0]
        prefixB = intentB.split("_")[0]
        
        if intentA == intentB:
            return 0  # 完全相同
        elif prefixA == prefixB:
            return 0.5  # 前缀相同但整体不同
        else:
            return 1  # 完全不同


    if labelA.startswith("SL:") and labelB.startswith("SL:"):
        slotA = labelA[3:]
        slotB = labelB[3:]
        return 0 if slotA==slotB else 0.5

    # 同为叶子 => embedding distance
    isLeafA = (not labelA.startswith("IN:")) and (not labelA.startswith("SL:"))
    isLeafB = (not labelB.startswith("IN:")) and (not labelB.startswith("SL:"))
    if isLeafA and isLeafB:
        return embedding_distance(labelA, labelB)*0.1

    # 否则 => 1
    return 1

def node_distance(nodeA: Node, nodeB: Node):
    """封装下, 供 zss.distance调用."""
    if nodeA is None or nodeB is None:
        return 1  # 如果出现空节点, cost=1
    return label_distance(nodeA.label, nodeB.label)

####################################
# 3. 计算zss distance & similarity
####################################

def mtop_tree_distance(parseA, parseB):
    treeA = parse_mtop_bracket(parseA)
    treeB = parse_mtop_bracket(parseB)

    # zss.distance 要求:
    # distance(A, B, get_children, insert_cost, remove_cost, update_cost)
    #   - insert_cost, remove_cost可以是常量或函数
    #   - update_cost = 函(nodeA, nodeB) -> cost
    dist = distance(
        treeA,
        treeB,
        get_children=lambda n: n.children,
        insert_cost=lambda n: 1,  # 插入节点 cost=1
        remove_cost=lambda n: 1,  # 删除节点 cost=1
        update_cost=node_distance # 替换节点 cost= node_distance(nodeA, nodeB)
    )
    return dist

def mtop_tree_similarity(parseA, parseB):
    dist = mtop_tree_distance(parseA, parseB)
    # 以 1/(1+dist) 返回 [0,1), 距离越小 相似度越大
    return 1/(1+dist)

####################################
# DEMO
####################################
def embedding_similarity(sentence1, sentence2):
    """
    计算两个句子的嵌入相似度。
    返回值为 0 到 1，1 表示完全相似。
    """
    # 计算句子嵌入
    embeddings = embedder.encode([sentence1, sentence2])
    vec1, vec2 = embeddings[0], embeddings[1]
    # 计算余弦相似度
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