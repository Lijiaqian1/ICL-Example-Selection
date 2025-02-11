import json
import argparse
import matplotlib.pyplot as plt
import os
from mtop_parse_sim import mtop_tree_similarity

def compute_per_item_stats(output_data, test_data):
    """
    返回一个 dict: key = test_idx or original_sentence, value = {
        "generated_parse": ...,
        "similarity": float,
        "exact_match": bool
    }
    这样你就可以逐条查看方法的性能
    """
    results_map = {}

    # 先把 test_data 做成一个从 (original_sentence) => gold_parse 的映射
    gold_map = {}
    for td in test_data:
        gold_map[td["original_sentence"]] = td["mtop_parsing"]

    for output_entry in output_data:
        idx = output_entry["idx"]
        gen_parse = output_entry["generated_parse"]
        sent = output_entry["original_sentence"]  # 这里假设 output里存了原句
        
        gold_parse = gold_map.get(sent, None)
        if gold_parse is None:
            # 可能找不到对应? 这里跳过
            continue
        
        sim = mtop_tree_similarity(gen_parse, gold_parse)
        exact = (gen_parse.strip() == gold_parse.strip())

        results_map[sent] = {
            "generated_parse": gen_parse,
            "similarity": sim,
            "exact_match": exact
        }
    return results_map

def compare_methods(methods_data):
    """
    methods_data: list of tuples (method_name, results_map), 
      where results_map is the output of compute_per_item_stats
    假设多个方法对应同一批句子 => 可比较 exact match分布
    返回: aggregated统计 + 各种对比
    """

    # 先确定所有的 sentence 集合
    all_sentences = set()
    for (mname, rmap) in methods_data:
        all_sentences.update(rmap.keys())

    # 统计
    method_stats = {}
    for (mname, rmap) in methods_data:
        # 计算overall exact match, avg sim
        total = len(rmap)
        exact_count = sum(1 for s in rmap if rmap[s]["exact_match"])
        avg_sim = 0.0
        if total>0:
            avg_sim = sum(rmap[s]["similarity"] for s in rmap)/total
        method_stats[mname] = {
            "total_count": total,
            "exact_count": exact_count,
            "exact_rate": exact_count/total if total>0 else 0,
            "avg_sim": avg_sim
        }

    # 如果只有2个方法，可以做更细的对比
    # 例如 venn-like: 
    #   sentences exactly matched by methodA only
    #   sentences exactly matched by methodB only
    #   sentences matched by both
    #   matched by none
    if len(methods_data)==2:
        mA_name, A_map = methods_data[0]
        mB_name, B_map = methods_data[1]
        
        both_exact = []
        onlyA = []
        onlyB = []
        none = []
        for sent in sorted(all_sentences):
            a_exact = (sent in A_map) and A_map[sent]["exact_match"]
            b_exact = (sent in B_map) and B_map[sent]["exact_match"]
            if a_exact and b_exact:
                both_exact.append(sent)
            elif a_exact and (not b_exact):
                onlyA.append(sent)
            elif b_exact and (not a_exact):
                onlyB.append(sent)
            else:
                none.append(sent)

        print(f"\n=== Detailed comparison for {mA_name} vs {mB_name} ===")
        print(f" Both exact match: {len(both_exact)}")
        print(f" Only {mA_name} exact match: {len(onlyA)}")
        print(f" Only {mB_name} exact match: {len(onlyB)}")
        print(f" Neither match: {len(none)}")

        # 也可以看看 whether there is a pattern in sentence length or structure
        # 这里示范仅打印数量

    return method_stats

def plot_exact_matches(method_stats):
    """
    对每个方法画一个bar: exact match rate
    """
    names = []
    rates = []
    for mname, stats in method_stats.items():
        names.append(mname)
        rates.append(stats["exact_rate"]*100.0)  # percentage

    plt.bar(names, rates, color='skyblue')
    plt.ylabel("Exact Match Rate (%)")
    plt.title("Exact Match Rate Comparison")
    for i, v in enumerate(rates):
        plt.text(i, v+0.5, f"{v:.1f}%", ha='center')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", required=True, help="Path to test JSON (with gold parse).")
    parser.add_argument("--result_files", nargs='+', required=True, 
                        help="One or more result JSON files to compare. e.g. embedding_results.json probe_results.json")
    args = parser.parse_args()

    # load test data
    with open(args.test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # load each result file, compute stats
    methods_data = []
    for rf in args.result_files:
        # infer method name from filename
        mname = os.path.splitext(os.path.basename(rf))[0]  # e.g. "embedding_results"
        with open(rf, 'r', encoding='utf-8') as ff:
            output_data = json.load(ff)
        # compute per-item
        rmap = compute_per_item_stats(output_data, test_data)
        methods_data.append((mname, rmap))

    # compare
    method_stats = compare_methods(methods_data)

    # print summary
    print("\n=== Overall Summary ===")
    for (mname, statsmap) in method_stats.items():
        print(f"{mname}: total={statsmap['total_count']}, exact_count={statsmap['exact_count']}, exact_rate={statsmap['exact_rate']:.3f}, avg_sim={statsmap['avg_sim']:.3f}")

    # optional: plot
    if len(method_stats)>1:
        plot_exact_matches(method_stats)


if __name__=="__main__":
    main()
