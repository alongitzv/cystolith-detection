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
print("-------------CANNABIS PREDICTOR-------------")
print("--------------------------------------------")

darknet_path = getcwd()
predictor_path = os.path.join(darknet_path, 'predictor')

print("Finding input images")

INPUT_IMAGES_PATH = os.path.join(predictor_path, 'input_images')
OUTPUT_PATH = os.path.join(predictor_path, 'output')

print("Calculating parameters for YOLO")

darknet_cmd_path = os.path.join(darknet_path, 'darknet')
obj_dat = os.path.join(darknet_path, 'data', 'obj.data')
yolo_cnf = os.path.join(darknet_path, 'cfg', 'yolov4-custom.cfg')
yolov4_path = os.path.dirname(darknet_path)
yolo_wghts = os.path.join(yolov4_path, 'training', 'yolov4-custom_best.weights')

im_paths = os.listdir(INPUT_IMAGES_PATH)
output_paths = os.listdir(OUTPUT_PATH)

cfgfilename = yolo_cnf.split('\\')[1].split('.')[0]
print(cfgfilename)

print("Loading CNN model")

# Clean output folder
for file in output_paths:
    file_path = os.path.join(OUTPUT_PATH, file)
    os.remove(file_path)

# Load json and create CNN model
json_file = open('cnn_model_files\model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("cnn_model_files\model1.h5")

print("Compiling CNN model from disk")

# Compile CNN loaded model

img_height, img_width = 224, 224
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

csv_output = []
# Create predictions for each image
for im in im_paths:
    if len(im) == 0:
        continue

    image_path = os.path.join(INPUT_IMAGES_PATH, im)
    image_name = (im.split('/')[-1]).split('.')[0]

    print("Predicting class via CNN model")

    # Predicting image via CNN model
    X = []
    if (os.path.isfile(image_path)):
        image = Image.open(image_path)
        img = image.resize((img_height, img_width))
        X.append(np.array(img) / 255.0)

    X = np.array(X)

    cnn_prediction_softmax = loaded_model.predict(X)
    cnn_prediction_class = np.argmax(cnn_prediction_softmax, axis=1)

    print("Predicting class via Yolo model")

    yolo_result_file_path = os.path.join(OUTPUT_PATH, image_name + '_yolo_result.txt')

    cmd = "%s detector test %s %s %s -dont_show -ext_output %s > %s" % (
    '"' + darknet_cmd_path + '"', '"' + obj_dat + '"', '"' + yolo_cnf + '"', '"' + yolo_wghts + '"',
    '"' + image_path + '"', '"' + yolo_result_file_path + '"')
    print('cmd = %s' % (cmd))
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    # wait
    p.wait(30)
    # make sure child process exit normally
    if p.poll() != 0:
        print("picture %s predict fails\n" % (im))
        break

    prediction_image_path = os.path.join(darknet_path, "predictions.jpg")
    os.rename(prediction_image_path, os.path.join(OUTPUT_PATH, image_name + '.jpg'))

    cystolith_cnt = 0
    max_cysto_percentage = -1
    fake_cystolith_cnt = 0
    max_fake_cysto_percentage = -1
    with open(yolo_result_file_path) as infile:
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

    print("CNN prediction : ", cnn_prediction_class[0], " (0-Real, 1-Synthetic)")

    print('Number of cystoliths detected: ', cystolith_cnt)

    print('Best accuracy of cystolith detection in %: ', max_cysto_percentage)

    print('Number of fake cystoliths detected: ', fake_cystolith_cnt)

    print('Best accuracy of fake cystolith detection in %: ', max_fake_cysto_percentage)

    prediction = ""
    print('PREDICTION: ')
    if (cystolith_cnt > 0 and cystolith_cnt >= (fake_cystolith_cnt*2) and max_cysto_percentage >= 50):
        print("Image name: ", image_name, ", Prediction: REAL CANNABIS")
        prediction = "REAL CANNABIS"
    elif (fake_cystolith_cnt > 0 and fake_cystolith_cnt >= (cystolith_cnt*2) and max_fake_cysto_percentage >= 50):
        print("Image name: ", image_name, ", Prediction: SYNTHETIC CANNABIS")
        prediction = "SYNTHETIC CANNABIS"
    elif (cnn_prediction_class[0]):
        print("Image name: ", image_name, ", Prediction: SYNTHETIC CANNABIS")
        prediction = "SYNTHETIC CANNABIS"
    else:
        print("Image name: ", image_name, ", Prediction: REAL CANNABIS")
        prediction = "REAL CANNABIS"

    if (max_cysto_percentage == -1):
        max_cysto_percentage = 0
    if (max_fake_cysto_percentage == -1):
        max_fake_cysto_percentage = 0

    csv_output.append([image_name, cnn_prediction_class[0], cystolith_cnt, max_cysto_percentage, fake_cystolith_cnt,
                       max_fake_cysto_percentage, prediction])

print("Creating a CSV output file")

with open( os.path.join(OUTPUT_PATH,'Predictor.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image name", "CNN Prediction (0-Real, 1-Synthetic)", "Real cystoliths YOLO detected",
                     "Best accuracy - cystolith detection in %", "Synthetic cystoliths YOLO detected",
                     "Best accuracy - synthetic cystolith detection in %", "Prediction"])
    writer.writerows(csv_output)

    # cp groud truth near the predictions dir
    # already draw box and saved in path_to_ground_truth dir, if not want to copy, just comment the next line
    # shutil.copy(path_to_ground_truth+'/'+image_name+'.jpg', path_to_save_predictions+'/'+image_name+"-truth.jpg")
    # print(image_name)
