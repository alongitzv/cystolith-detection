# This Python code converts a dataset in YOLO format into the COCO format.
# The YOLO dataset contains images of bottles and the bounding box annotations in the
# YOLO format. The COCO format is a widely used format for object detection datasets.

# The input and output directories are specified in the code. The categories for
# the COCO dataset are also defined, with only one category for "bottle". A dictionary for the COCO dataset is initialized with empty values for "info", "licenses", "images", and "annotations".

# The code then loops through each image in the input directory. The dimensions
# of the image are extracted and added to the COCO dataset as an "image" dictionary,
# including the file name and an ID. The bounding box annotations for each image are
# read from a text file with the same name as the image file, and the coordinates are
# converted to the COCO format. The annotations are added to the COCO dataset as an
# "annotation" dictionary, including an ID, image ID, category ID, bounding box coordinates,
# area, and an "iscrowd" flag.

# The COCO dataset is saved as a JSON file in the output directory.

import json
import os
from PIL import Image


def yolo_to_detr_annotations(input_dir, output_dir):
    # Define the categories for the COCO dataset
    categories = [{"id": 0, "name": "real"}, {"id": 1, "name": "fake"}]

    # Define the COCO dataset dictionary
    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    #train_list_path = r'D:\Users Data\arthurSoussan\Desktop\detr\datasets_lists\val.txt'
    #with open(train_list_path, 'r') as file:
        # Read the entire content of the file as a single string
    #    content = file.read()

    # Loop through the images in the input directory
    for image_file in os.listdir(input_dir):
        if 'txt' in image_file:
            continue

        #if image_file not in content:
        #    continue

        # Load the image and get its dimensions
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path)
        width, height = image.size

        # Add the image to the COCO dataset
        image_dict = {
            "id": image_file.split('.')[0],
            "width": width,
            "height": height,
            "file_name": image_file
        }
        coco_dataset["images"].append(image_dict)

        # Load the bounding box annotations for the image
        with open(os.path.join(input_dir, f'{image_file.split(".")[0]}.txt')) as f:
            annotations = f.readlines()

        # Loop through the annotations and add them to the COCO dataset
        for ann in annotations:
            x, y, w, h = map(float, ann.strip().split()[1:])
            x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
            x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
            ann_dict = {
                "id": len(coco_dataset["annotations"]),
                "image_id": image_file.split('.')[0],
                "category_id": 0,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0
            }
            coco_dataset["annotations"].append(ann_dict)

    # Save the COCO dataset to a JSON file
    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(coco_dataset, f)


# Set the paths for the input and output directories
train_input_dir = r'D:\Users Data\arthurSoussan\Desktop\detr\train_images_and_anno_split_2'
train_output_dir = r'D:\Users Data\arthurSoussan\Desktop\detr\train_images_and_anno_split_2'
val_input_dir = r'D:\Users Data\arthurSoussan\Desktop\detr\val_images_and_anno_split_2'
val_output_dir = r'D:\Users Data\arthurSoussan\Desktop\detr\val_images_and_anno_split_2'
test_input_dir = r'D:\Users Data\arthurSoussan\Desktop\detr\test_images_and_anno_split_2'
test_output_dir = r'D:\Users Data\arthurSoussan\Desktop\detr\test_images_and_anno_split_2'

yolo_to_detr_annotations(train_input_dir, train_output_dir)
yolo_to_detr_annotations(val_input_dir, val_output_dir)
yolo_to_detr_annotations(test_input_dir, test_output_dir)