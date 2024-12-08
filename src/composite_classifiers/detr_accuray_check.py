import os
from os import getcwd
import subprocess
import time
import shutil
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
import csv

print("--------------------------------------------")
print("-------------DETR ACCURACY CHECK-------------")
print("--------------------------------------------")

INPUT_IMAGES_TEXT_OUTPUT__PATH = r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\detr_test_results"
OUTPUT_PATH = r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\detr_statistics"

contents = os.listdir(INPUT_IMAGES_TEXT_OUTPUT__PATH)
csv_output = []
# Print the contents of the directory
for im in contents:
    if ".jpg" in im:
        continue
    image_name = (im.split('/')[-1]).split('.')[0]
    print("Image - ", image_name)

    detr_result_file_path = os.path.join(INPUT_IMAGES_TEXT_OUTPUT__PATH, im)

    cystolith_cnt = 0
    max_cysto_percentage = -1
    fake_cystolith_cnt = 0
    max_fake_cysto_percentage = -1
    with open(detr_result_file_path) as infile:
        for line in infile:
            if 'cystolith' in line:
                if 'fake' not in line:
                    cystolith_cnt += 1
                    percentage = int(line.split('%')[0][-2:])
                    if (percentage > max_cysto_percentage):
                        max_cysto_percentage = percentage
            if 'fake_cystolith' in line:
                fake_cystolith_cnt += 1
                percentage = int(line.split('%')[0][-2:])
                if (percentage > max_fake_cysto_percentage):
                    max_fake_cysto_percentage = percentage


    print('Number of cystoliths detected: ', cystolith_cnt)

    print('Best accuracy of cystolith detection in %: ', max_cysto_percentage)

    print('Number of fake cystoliths detected: ', fake_cystolith_cnt)

    print('Best accuracy of fake cystolith detection in %: ', max_fake_cysto_percentage)

    prediction = ""
    print('PREDICTION: ')
    if (cystolith_cnt > fake_cystolith_cnt):
        print("Image name: ", image_name, ", Prediction: REAL CANNABIS")
        prediction = "REAL CANNABIS"
    elif (fake_cystolith_cnt > cystolith_cnt):
        print("Image name: ", image_name, ", Prediction: SYNTHETIC CANNABIS")
        prediction = "SYNTHETIC CANNABIS"
    elif (max_cysto_percentage > max_fake_cysto_percentage):
        print("Image name: ", image_name, ", Prediction: REAL CANNABIS")
        prediction = "REAL CANNABIS"
    elif (max_fake_cysto_percentage > max_cysto_percentage):
        print("Image name: ", image_name, ", Prediction: SYNTHETIC CANNABIS")
        prediction = "SYNTHETIC CANNABIS"
    else :
        print("Image name: ", image_name, ", Prediction: NOT DETECTED")
        prediction = "NOT DETECTED"

    if (max_cysto_percentage == -1):
        max_cysto_percentage = 0
    if (max_fake_cysto_percentage == -1):
        max_fake_cysto_percentage = 0

    csv_output.append([image_name, cystolith_cnt, max_cysto_percentage, fake_cystolith_cnt,
                       max_fake_cysto_percentage, prediction])

print("Creating a CSV output file")

with open( os.path.join(OUTPUT_PATH,'detr_accuracy_check_2705_split_1_0.01_0.01.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image name", "Real cystoliths DETR detected",
                     "Best accuracy - cystolith detection in %", "Synthetic cystoliths DETR detected",
                     "Best accuracy - synthetic cystolith detection in %", "Prediction"])
    writer.writerows(csv_output)

