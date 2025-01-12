import numpy as np
import argparse

def main(file_path):
    data = np.load(file_path, allow_pickle=True)
    print("Keys in the .npz file:", data.files)

    for key in data.files:
        print(f"\nKey: {key}")
        
        try:
            print("Array shape:", data[key].shape)
            print("Array dtype:", data[key].dtype)

            array = data[key]
            if array.ndim == 1:  
                print("Array data (first 10 items):", array[:10])
            elif array.ndim == 2:  
                print("Array data (first 5 rows):")
                print(array[:5])  
            else:  
                print("Array data (first slice):")
                print(array[0]) 
        except Exception as e:
            print(f"Error loading key '{key}': {e}")

    print("\nNote: Only a small portion of each array is displayed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display the contents of a .npz file.")
    parser.add_argument("file_path", type=str, help="Path to the .npz file")
    args = parser.parse_args()

    main(args.file_path)
