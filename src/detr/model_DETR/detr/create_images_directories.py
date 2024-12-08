import os
import shutil

def create_directories(txt_files, source_dir, output_dirs):
    # Create output directories if they don't exist
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)

    # Iterate through each txt file
    for txt_file, output_dir in zip(txt_files, output_dirs):
        with open(txt_file, 'r') as f:
            # Read lines from the txt file
            lines = f.readlines()
            # Iterate through each line (assuming each line contains a filename)
            for line in lines:
                # Strip newline characters and construct the full path
                filename = line.strip().replace("data/obj/", "")
                txt_filename = filename[:-4]+".txt"
                source_file = os.path.join(source_dir, filename)
                txt_source_file = os.path.join(source_dir, txt_filename)
                # Check if the file exists and if it's a file (not a directory)
                if os.path.exists(source_file) and os.path.isfile(source_file):
                    # Move the file to the corresponding output directory
                    shutil.copy(source_file, output_dir)
                    shutil.copy(txt_source_file, output_dir)
                else:
                    print(f"File {filename} does not exist or is not a file.")

# Define paths and filenames
txt_files = ["./datasets_lists/train_split_2.txt", "./datasets_lists/val_split_2.txt", "./datasets_lists/test_split_2.txt"]
source_dir = r'D:\Users Data\arthurSoussan\Desktop\yolov4\darknet\data\obj'
output_dirs = ["./train_images_and_anno_split_2", "./val_images_and_anno_split_2", "./test_images_and_anno_split_2"]

# Call the function
create_directories(txt_files, source_dir, output_dirs)