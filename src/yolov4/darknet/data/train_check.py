import os


def filter_images(train_file, folder_path, output_file):
    # Read train.txt file
    with open(train_file, 'r') as f:
        train_paths = f.readlines()

    # Get list of image files in the specified folder
    image_files = os.listdir(folder_path)

    # Filter paths from train.txt that exist in the folder
    filtered_paths = [path.strip() for path in train_paths if os.path.basename(path.strip()) in image_files]

    # Write filtered paths to new_test.txt
    with open(output_file, 'w') as f:
        for path in filtered_paths:
            f.write(path + '\n')


if __name__ == "__main__":
    train_file = "train.txt"
    folder_path = r"D:\Users Data\arthurSoussan\Desktop\yolov4\darknet\data\obj"
    output_file = "new_test.txt"

    filter_images(train_file, folder_path, output_file)
    print("Filtered paths written to new_test.txt")