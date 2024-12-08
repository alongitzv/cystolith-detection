import os


def read_directories_from_file(file_path):
    updated_directories = []
    with open(file_path, 'r') as file:
        directories = file.readlines()
        # Modify directory paths
        for line in directories:
            parts = line.strip().split('/')
            raw_path = parts[-1]
            updated_path = "data/obj/"+raw_path+"\n"
            updated_directories.append(updated_path)
    print("FILE - "+file_path+" - DONE")
    return updated_directories


def process_image_directories(file_paths):
    updated_files = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            updated_file = read_directories_from_file(file_path)
            updated_files.append(updated_file)
        else:
            print(f"File not found: {file_path}")
    return updated_files


def write_updated_files(file_paths, updated_files):
    for idx, file_path in enumerate(file_paths):
        output_file = r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\split_2_sets\updated_file_"+str(idx + 1)+".txt"
        with open(output_file, 'w') as file:
            file.writelines(updated_files[idx])
        print(f"Updated file saved as: {output_file}")


if __name__ == "__main__":
    # Replace these with the paths to your text files
    file_paths = [r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\split_2_sets\val_split_2.txt",
                  r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\split_2_sets\test_split_2.txt",
                  r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\split_2_sets\train_split_2.txt"]

    # Process image directories
    updated_files = process_image_directories(file_paths)

    # Write updated files
    write_updated_files(file_paths, updated_files)