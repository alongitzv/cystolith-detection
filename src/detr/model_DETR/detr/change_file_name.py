import json

def update_ids(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create a mapping from old ids to new ids
    for image in data['images']:
        image['file_name'] = image['file_name'].replace(" ","")

    # Write the updated JSON back to the file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

# Path to the JSON file
json_file1 =  r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\val_split_2\annotations.json"
json_file2 = r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\test_split_2\annotations.json"
json_file3 =  r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\train_split_2\annotations.json"


# Call the function to update ids
update_ids(json_file1)
update_ids(json_file2)
update_ids(json_file3)