import os

tested_images_path = r"D:\Users Data\arthurSoussan\Desktop\yolov4\darknet\yolo_accuracy_check_results\output_2705_all_images_split_1"
yolo_obj = r"D:\Users Data\arthurSoussan\Desktop\yolov4\darknet\data\obj"



for root, dirs, files in os.walk(yolo_obj):
    # Iterate through the files in each subfolder
    for file in files:
        # Check if the file is an image file (you may adjust this check based on your image formats)
        if ".txt" in file:
            continue
        image_path = os.path.join(root, file)
        was_found = False
        for root2, dirs2, files2 in os.walk(tested_images_path):
            # Iterate through the files in each subfolder
            for file2 in files2:
                # Check if the file is an image file (you may adjust this check based on your image formats)
                if ".txt" in file2:
                    continue
                if file == file2:
                    was_found = True
                    break
        if was_found==False:
            print(image_path)

