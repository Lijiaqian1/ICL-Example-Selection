'''import os
import penman
import json

def load_amr_from_file(file_path):
    """
    读取一个 txt 文件中的所有 AMR，返回一个列表，每个元素可包含:
    {
      "id": <str>,
      "sentence": <str>,
      "graph": <str>  # 将 Graph 转换为字符串
    }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        graphs = penman.load(f)

    items = []
    for g in graphs:
        amr_id = g.metadata.get('id', None)
        sent = g.metadata.get('snt', None)
        # 将 Graph 对象编码为字符串
        graph_str = penman.encode(g)
        items.append({
            "id": amr_id,
            "sentence": sent,
            "graph": graph_str
        })
    return items

def load_all_amrs(folder_path):
    """
    遍历 folder_path 下的所有 txt 文件，合并所有 AMR 到一个列表返回。
    """
    all_data = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.txt'):
            fpath = os.path.join(folder_path, fname)
            data_part = load_amr_from_file(fpath)
            all_data.extend(data_part)
    return all_data

if __name__ == "__main__":
    #folder = "../data/amrs/split/demo"  
    folder = "../data/amrs/split/training" 
    all_amrs = load_all_amrs(folder)
    print(f"Total AMRs loaded: {len(all_amrs)}")
    
    # 保存到 JSON 文件
    with open("../data/amrs/split/training/all_amrs.json", "w", encoding="utf-8") as fw:
        json.dump(all_amrs, fw, ensure_ascii=False, indent=2)'''
import os
import penman
import json

def load_amr_from_file(file_path):
    """
    读取一个 txt 文件中的所有 AMR，返回一个列表，每个元素可包含:
    {
      "id": <str>,
      "sentence": <str>,
      "graph": <str>  # 将 Graph 转换为字符串
    }
    """
    items = []
    current_amr_lines = []
    current_metadata = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# ::'):
                # 提取注释行中的元数据
                if line.startswith('# ::id'):
                    current_metadata['id'] = line.split(' ', 2)[-1].strip()
                elif line.startswith('# ::snt'):
                    current_metadata['sentence'] = line.split(' ', 2)[-1].strip()
            elif line == "":
                # 如果遇到空行，处理当前AMR图
                if current_amr_lines:
                    try:
                        graph = penman.decode("\n".join(current_amr_lines))
                        graph_str = penman.encode(graph)
                        items.append({
                            "id": current_metadata.get('id', None),
                            "sentence": current_metadata.get('sentence', None),
                            "graph": graph_str
                        })
                    except Exception as e:
                        print(f"Error decoding AMR in file {file_path}: {e}")
                    finally:
                        # 重置当前AMR
                        current_amr_lines = []
                        current_metadata = {}
            else:
                # 将AMR图行添加到当前AMR中
                current_amr_lines.append(line)

        # 文件末尾的最后一个AMR
        if current_amr_lines:
            try:
                graph = penman.decode("\n".join(current_amr_lines))
                graph_str = penman.encode(graph)
                items.append({
                    "id": current_metadata.get('id', None),
                    "sentence": current_metadata.get('sentence', None),
                    "graph": graph_str
                })
            except Exception as e:
                print(f"Error decoding final AMR in file {file_path}: {e}")

    return items

def load_all_amrs(folder_path):
    """
    遍历 folder_path 下的所有 txt 文件，合并所有 AMR 到一个列表返回。
    """
    all_data = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.txt'):
            fpath = os.path.join(folder_path, fname)
            data_part = load_amr_from_file(fpath)
            all_data.extend(data_part)
    return all_data

if __name__ == "__main__":
    # folder = "../data/amrs/split/demo"  
    folder = "../data/amrs/split/training" 
    all_amrs = load_all_amrs(folder)
    print(f"Total AMRs loaded: {len(all_amrs)}")

    # 保存到 JSON 文件
    with open("../data/amrs/split/training/all_amrs.json", "w", encoding="utf-8") as fw:
        json.dump(all_amrs, fw, ensure_ascii=False, indent=2)
