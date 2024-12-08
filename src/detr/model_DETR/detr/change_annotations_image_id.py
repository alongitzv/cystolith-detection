import json

def update_image_ids(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create a mapping from image filenames to image ids
    image_filename_to_id = {image['file_name']: image['id'] for image in data['images']}
    print(image_filename_to_id)

    # Update the "image_id" field in the annotations
    for annotation in data['annotations']:
        image_filename = annotation['image_id']
        print("image id", image_filename)
        image_filename = image_filename + ".jpg"
        print("image_filename", image_filename)
        if image_filename in image_filename_to_id:
            annotation['image_id'] = image_filename_to_id[image_filename]
        else:
            print(r"Image filename ",image_filename," not found in images.")

    # Write the updated JSON back to the file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

# Path to the JSON file
json_file = r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\train_split_2\annotations.json"
# Call the function to update image ids
update_image_ids(json_file)
print("train- done")

# Path to the JSON file
json_file = r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\val_split_2\annotations.json"
# Call the function to update image ids
update_image_ids(json_file)
print("val- done")

# Path to the JSON file
json_file=r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\test_split_2\annotations.json"
# Call the function to update image ids
update_image_ids(json_file)
print("test- done")
