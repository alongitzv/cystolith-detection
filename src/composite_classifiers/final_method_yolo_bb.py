
import os
import torch
import roboflow
import supervision
import transformers
from transformers import DetrForObjectDetection, DetrImageProcessor
import pytorch_lightning
import torchvision
import cv2
from torch.utils.data import DataLoader
import csv
import os.path
import argparse
import yaml
import shutil
import torch.utils.data
import subprocess

# separate py files
from utils import *
from models import *
from data_loader import *



"""#########################                                    FINAL METHOD                                        #########################"""
"""#########################               YOLO threshold = 0.05  -->  YOLO 0.01  -->  CNN BOUNDING BOXED           #########################"""


# settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
ANNOTATION_FILE_NAME = "annotations.json"
YOLO_WEIGHTS_FOLDER = "BACKUP_BATCHES_WINTER_2024\yolo_training_5_split_2_2305"
#YOLO_WEIGHTS_FOLDER = "BACKUP_BATCHES_WINTER_2024\yolo_training_4_split_1_0205"
CNN_WHOLE_MODEL = "./CNN_models/cfg_04_ETAZ_f2_1a/cysto_net.pth"
#CNN_WHOLE_MODEL = "./CNN_models/cfg_04_ETAZ_f1_1a/cysto_net.pth"
CNN_BB_MODEL = "./CNN_models/cfg_06_f2_1b_g7_bounding_boxes/cysto_net.pth"
#CNN_BB_MODEL = "./CNN_models/cfg_06_f1_1d_bounding_boxes/cysto_net.pth"
bb_images_path = "./bb_images_temp"
csv_path = "./final_method_results/final_method_yolo_bb_prediction_split_2.csv"
image_directory_path = r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\test_split_2" # images folder - can be changed by user
INPUT_TEST_LIST = os.path.join(r"D:\Users Data\arthurSoussan\Desktop\Cannabis_project_2024\dataset_and_splits\splits\lists\split_2\without_spaces", 'test_split_2.txt')
#INPUT_TEST_LIST = "./simulation_test_split_1.txt"
OUTPUT_PATH = "./yolo_image_temp"
temp_list = "./image_to_yolo_predict.txt"
DARKNET_PATH = r"D:\Users Data\arthurSoussan\Desktop\yolov4\darknet"

def cnn_predict(img_file='', ann_file='', model='', type=''):  #returns CNN detections on single image

    print("Start CNN inference")

    # initiate params
    img_list = []
    do_resize = []
    do_bbx = False
    if type == 'whole':
        do_resize = [214,512]
    elif type == 'yolo' or type == 'detr':
        do_resize = [256,256]
        do_bbx = True


    # specify device, CUDA if available, CPU otherwise
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load the specified model
    net = DLA(input_size=do_resize)
    net.to(device)
    state = torch.load(model)
    net.load_state_dict(state['model_state'])
    net.eval()

    # create yolo/detr bbx
    if do_bbx == True:
        # load original image and yolo/detr output file
        # extract bounding boxes and save them as separate images in a temp folder
        img_list = classifier_bbx_to_images_single_pic(img_file=img_file, ann_file=ann_file, out_dir=bb_images_path, out_size=[512, 512], classifier=type)
    else:
        img_list = [img_file]

    num_images = len(img_list)
    res = np.zeros((num_images,3), dtype=float)
    softmax = nn.Softmax(dim=1)

    # no gradients needed
    with torch.no_grad():
        for i in range(0,num_images):
            curr_img = img_list[i]
            img = imageio.v2.imread(curr_img)
            if do_resize[0] > 0:
                img = Image.fromarray(img).resize((do_resize[1], do_resize[0]), resample=Image.BICUBIC)
                img = np.array(img)
            img = np.array(img, dtype=float)     # img = np.array(img, dtype=np.float)   # img = np.array(img, dtype=float)
            img = img.astype(float) / 255.0      # img = img.astype(np.float) / 255.0    # img = img.astype(float) / 255.0   

            img = img.transpose(2, 0, 1)  # NHWC -> NCHW
            img = np.expand_dims(img, 0)  # needed to to reshape CHW -> NCHW, N=1
            img = torch.from_numpy(img).float()

            img = img.to(device)
            outputs = net(img)
            _, prediction = torch.max(outputs, 1)


    print("End CNN inference")
    if num_images == 0: # zero detections after second detr run
        return -1
    else:
        return int(prediction.item())


def yolo_predict(weights_folder, threshold, input_list):

    csv_output = []

    print("Calculating parameters for YOLO")
    darknet_path = DARKNET_PATH
    darknet_cmd_path = os.path.join(darknet_path, 'darknet')
    obj_dat = os.path.join(darknet_path, 'data', 'obj.data')
    yolo_cnf = os.path.join(darknet_path, 'cfg', 'yolov4-custom.cfg')
    yolov4_path = os.path.dirname(darknet_path)
    yolo_wghts = os.path.join(yolov4_path, weights_folder , 'yolov4-custom_best.weights')  # update weights according to used model

    im_paths = os.listdir(image_directory_path)
    output_paths = os.listdir(OUTPUT_PATH)

    cfgfilename = yolo_cnf.split('\\')[1].split('.')[0]
    print(cfgfilename)

    # Create predictions for each image
    with open( input_list, 'r') as file:
        for im in file:
            im = im.strip()
            im = im[len("data/obj/"):]  # Remove the prefix

            if len(im) == 0:
                continue

            image_path = os.path.join(image_directory_path, im)
            image_name = (im.split('/')[-1]).split('.')[0]

            print("Predicting class via Yolo model")

            yolo_result_file_path = os.path.join(OUTPUT_PATH, image_name + '_yolo_result_'+ str(threshold)+ '.txt')

            cmd = "%s detector test %s %s %s -dont_show -ext_output %s -thresh %s > %s" % (
            '"' + darknet_cmd_path + '"', '"' + obj_dat + '"', '"' + yolo_cnf + '"', '"' + yolo_wghts + '"',
            '"' + image_path + '"', threshold, '"' + yolo_result_file_path + '"')
            print('cmd = %s' % (cmd))
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            # wait
            p.wait(30)
            # make sure child process exit normally
            if p.poll() != 0:
                print("picture %s predict fails\n" % (im))
                break

            prediction_image_path = "./predictions.jpg" # os.path.join(darknet_path, "predictions.jpg")
            os.rename(prediction_image_path, os.path.join(OUTPUT_PATH, image_name +"_"+ str(threshold) + '.jpg'))
            csv_image_result = yolo_decision_strategy(yolo_result_file_path, image_path, image_name, threshold)
            csv_output.append(csv_image_result)
    return csv_output


def yolo_decision_strategy (yolo_result_file_path, image_path, image_name, threshold): #return array with results ready for csv\

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

    print('Number of cystoliths detected: ', cystolith_cnt)
    print('Best accuracy of cystolith detection in %: ', max_cysto_percentage)
    print('Number of fake cystoliths detected: ', fake_cystolith_cnt)
    print('Best accuracy of fake cystolith detection in %: ', max_fake_cysto_percentage)

    prediction = ""
    if (cystolith_cnt > fake_cystolith_cnt):
        prediction = "REAL CANNABIS"
    elif (fake_cystolith_cnt > cystolith_cnt):
        prediction = "SYNTHETIC CANNABIS"
    elif (max_cysto_percentage > max_fake_cysto_percentage):
        prediction = "REAL CANNABIS"
    elif (max_fake_cysto_percentage > max_cysto_percentage):
        prediction = "SYNTHETIC CANNABIS"
    else :  
        prediction = "NOT DETECTED"
        
        if threshold == 0.05:
            txt_file_content = "data/obj/"+image_name+".jpg"
            with open(temp_list, 'w') as file:
                file.write(txt_file_content)
            lower_threshold_csv_output = yolo_predict(YOLO_WEIGHTS_FOLDER, 0.01, temp_list)
            ann = OUTPUT_PATH + "/"+ image_name + '_yolo_result_0.01.txt'
            cnn_bb_prediction = cnn_predict(img_file=image_path, ann_file=ann, model=CNN_BB_MODEL, type='yolo')
            
            if cnn_bb_prediction == 0:
                prediction = "CNN BB - REAL CANNABIS"
            elif cnn_bb_prediction == 1:
                prediction = "CNN BB - SYNTHETIC CANNABIS"
    
            else: # threshold == 0.01
                cnn_whole_prediction = cnn_predict(img_file=image_path, ann_file='', model=CNN_WHOLE_MODEL, type='whole')

                if cnn_whole_prediction == 0:
                    prediction = "CNN WHOLE - REAL CANNABIS"
                elif cnn_whole_prediction == 1:
                    prediction = "CNN WHOLE - SYNTHETIC CANNABIS"

    print("Image name: ", image_name, ", Prediction: ", prediction)

    if (max_cysto_percentage == -1):
        max_cysto_percentage = 0
    if (max_fake_cysto_percentage == -1):
        max_fake_cysto_percentage = 0

    return [image_name, cystolith_cnt, max_cysto_percentage, fake_cystolith_cnt, max_fake_cysto_percentage, prediction]


# Test model


###################################  YOLO PREDICTION + DECISION STRATEGY + CNN  ###################################

csv_output = yolo_predict(YOLO_WEIGHTS_FOLDER, 0.05, INPUT_TEST_LIST)

print("Creating a CSV output file")

with open( csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image name", "Real cystoliths DETR detected",
                     "Best accuracy - cystolith detection in %", "Synthetic cystoliths DETR detected",
                     "Best accuracy - synthetic cystolith detection in %", "Prediction"])
    writer.writerows(csv_output)


