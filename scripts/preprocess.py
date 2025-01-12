
import os
import penman
import json

def load_amr_from_file(file_path):

    items = []
    current_amr_lines = []
    current_metadata = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# ::'):
                if line.startswith('# ::id'):
                    current_metadata['id'] = line.split(' ', 2)[-1].strip()
                elif line.startswith('# ::snt'):
                    current_metadata['sentence'] = line.split(' ', 2)[-1].strip()
            elif line == "":
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
                        current_amr_lines = []
                        current_metadata = {}
            else:         
                current_amr_lines.append(line)

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

    with open("../data/amrs/split/training/all_amrs.json", "w", encoding="utf-8") as fw:
        json.dump(all_amrs, fw, ensure_ascii=False, indent=2)
