import numpy as np
import argparse

def main(file_path):
    # 使用 allow_pickle=True 加载数据，解决无法加载 object 数组的问题
    data = np.load(file_path, allow_pickle=True)

    # 查看文件中所有的键
    print("Keys in the .npz file:", data.files)

    # 查看每个键对应的数据（只显示部分内容）
    for key in data.files:
        print(f"\nKey: {key}")
        
        try:
            print("Array shape:", data[key].shape)
            print("Array dtype:", data[key].dtype)
            
            # 限制显示前几项数据
            array = data[key]
            if array.ndim == 1:  # 一维数组
                print("Array data (first 10 items):", array[:10])
            elif array.ndim == 2:  # 二维数组
                print("Array data (first 5 rows):")
                print(array[:5])  # 显示前5行
            else:  # 高维数组
                print("Array data (first slice):")
                print(array[0])  # 显示第一片
        except Exception as e:
            print(f"Error loading key '{key}': {e}")

    print("\nNote: Only a small portion of each array is displayed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display the contents of a .npz file.")
    parser.add_argument("file_path", type=str, help="Path to the .npz file")
    args = parser.parse_args()

    main(args.file_path)
