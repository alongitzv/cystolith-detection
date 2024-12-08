import json

def process_json(input_file):
    with open(input_file, 'r+') as f:
        data = json.load(f)

        for image in data['images']:
            if 'S' in image['file_name']:
                image_id = image['id']
                for annotation in data['annotations']:
                    if annotation['image_id'] == image_id:
                        annotation['category_id'] = 1

        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

    print(f"Modified JSON saved to {input_file}")

if __name__ == "__main__":
    input_file1 = r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\test_split_2\annotations.json"
    input_file2 = r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\train_split_2\annotations.json"
    input_file3 = r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\val_split_2\annotations.json"
    process_json(input_file1)
    process_json(input_file2)
    process_json(input_file3)