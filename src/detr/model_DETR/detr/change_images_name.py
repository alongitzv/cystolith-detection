import os

def remove_spaces_from_filenames(directory):
    for filename in os.listdir(directory):
        if " " in filename:
            new_filename = filename.replace(" ", "")  # Replace space with underscore
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(r"Renamed '{filename}' to '{new_filename}'")

# Define paths to the three directories
directories = [r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\test_split_2",
               r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\val_split_2",
               r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\train_split_2"]

# Iterate over each directory and remove spaces from filenames
for directory in directories:
    remove_spaces_from_filenames(directory)