
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

# separate py files
from utils import *
from models import *
from data_loader import *



"""#########################                                    FINAL METHOD                               #########################"""
"""#########################                            DETR 0.3 & 0.5  -->  CNN WHOLE                     #########################"""


# settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
ANNOTATION_FILE_NAME = "annotations.json"
DETR_MODEL_PATH = "./DETR_models/detr_model_split2"
CNN_WHOLE_MODEL = "./CNN_models/cfg_04_ETAZ_f2_1a/cysto_net.pth"
csv_path = "./final_method_results/final_method_whole_prediction_split_2.csv"
image_directory_path = r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\test_split_2" # images folder - can be changed by user



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


def detr_predict(model, image, confidence_threshold, detection_threshold ): #returns DETR detections on single image
    with torch.no_grad():
        print("Load images and predict")
        # load image and predict
        inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
        outputs = model(**inputs)

        # post-process
        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=confidence_threshold,
            target_sizes=target_sizes
        )[0]
            
    return supervision.Detections.from_transformers(transformers_results=results).with_nms(threshold=detection_threshold) # detr detections


def detr_decision_strategy (detections, image_name ): #return array with results ready for csv
    cystolith_cnt = 0
    max_cysto_percentage = -1
    fake_cystolith_cnt = 0
    max_fake_cysto_percentage = -1

    for class_id, confidence in zip(detections.class_id, detections.confidence):
            if class_id == 0: # real
                cystolith_cnt += 1
                percentage = int(confidence * 100)
                if (percentage > max_cysto_percentage):
                    max_cysto_percentage = percentage
            if class_id == 1: # fake
                fake_cystolith_cnt += 1
                percentage = int(confidence * 100)
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

    print("Image name: ", image_name, ", Prediction: ", prediction)

    if (max_cysto_percentage == -1):
        max_cysto_percentage = 0
    if (max_fake_cysto_percentage == -1):
        max_fake_cysto_percentage = 0

    return [image_name, cystolith_cnt, max_cysto_percentage, fake_cystolith_cnt, max_fake_cysto_percentage, prediction]


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        image_processor,
        train: bool = True
    ):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

def collate_fn(batch):
    # DETR authors employ various image sizes during training, making it not possible
    # to directly batch together images. Hence they pad the images to the biggest
    # resolution in a given batch, and create a corresponding binary pixel_mask
    # which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
model = DetrForObjectDetection.from_pretrained(DETR_MODEL_PATH)
model.to(DEVICE)

TEST_DATASET = CocoDetection(
    image_directory_path= image_directory_path,
    image_processor=image_processor,
    train=False)

# we will use id2label function
categories = TEST_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}

class Detr(pytorch_lightning.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here:
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER


# Test model

csv_output = []

for root, dirs, files in os.walk(image_directory_path):
    # Iterate through the files in each subfolder
    for file in files:     
        if ".json" in file:
            continue
        image_path = os.path.join(root, file)
        image = cv2.imread(image_path)
        image_name = file.split('.')[0]


        ###################################  DETR PREDICTION  ###################################

        detections = detr_predict(model, image, 0.3, 0.5)
           
        ################################### DECISION STRATEGY ###################################

        csv_image_result = detr_decision_strategy(detections, image_name)

        if csv_image_result[-1] == "NOT DETECTED":
        ###################################  CNN PREDICTION  ###################################

            cnn_whole_prediction = cnn_predict(img_file=image_path, ann_file='', model=CNN_WHOLE_MODEL, type='whole')
            if cnn_whole_prediction == 0:
                csv_image_result[-1] = "CNN WHOLE - REAL CANNABIS"
            elif cnn_whole_prediction == 1:
                csv_image_result[-1] = "CNN WHOLE - SYNTHETIC CANNABIS"

        csv_output.append(csv_image_result)


print("Creating a CSV output file")

with open( csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image name", "Real cystoliths DETR detected",
                     "Best accuracy - cystolith detection in %", "Synthetic cystoliths DETR detected",
                     "Best accuracy - synthetic cystolith detection in %", "Prediction"])
    writer.writerows(csv_output)


