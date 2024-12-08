
import os

HOME = os.getcwd()


import torch

# !nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

import roboflow
import supervision
import transformers
import pytorch_lightning


print(
    "roboflow:", roboflow.__version__,
    "; supervision:", supervision.__version__,
    "; transformers:", transformers.__version__,
    "; pytorch_lightning:", pytorch_lightning.__version__
)

"""## Inference with pre-trained COCO model

### Download Data
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
# !wget https://media.roboflow.com/notebooks/examples/dog.jpeg

# IMAGE_NAME = "youtube_man.jpg"  # "C (2).jpg"
# IMAGE_PATH = os.path.join(HOME, IMAGE_NAME)

"""### Load Model"""

import torchvision

from transformers import DetrForObjectDetection, DetrImageProcessor

# settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
CONFIDENCE_THRESHOLD = 0.01
IOU_TRESHOLD = 0.8

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
model.to(DEVICE)

"""### Inference"""
#
# Commented out IPython magic to ensure Python compatibility.
import cv2
import supervision as sv

ANNOTATION_FILE_NAME = "annotations.json"


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


from torch.utils.data import DataLoader

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

TEST_DATASET = CocoDetection(
    image_directory_path= r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\test_2024",
    image_processor=image_processor,
    train=False)

print("Number of test examples:", len(TEST_DATASET))


TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=4)


# we will use id2label function for training
categories = TEST_DATASET.coco.cats
print("categories- ", categories)
id2label = {k: v['name'] for k,v in categories.items()}
print("id2label:\n", id2label)


import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch


class Detr(pl.LightningModule):

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

"""**NOTE:** Let's start `tensorboard`."""



# CHECKPOINT = r"detr_saved_model\lightning_logs\version_0\checkpoints\epoch=1401-step=32246.ckpt"

image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
# model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
#
print("after checkpoint")
#MODEL_PATH = os.path.join(HOME, 'final_sets/detr_saved_model/detr-custom-model')
MODEL_PATH = r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\old_detr_models\500_epochs_final_set_0105_split1"
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)

# print("after loading")
model.to(DEVICE)

# model = Detr.load_from_checkpoint(CHECKPOINT, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
# print(model.learning_rate)

""" INITIAL CHECK THAT ANNOATIONS ARE SHOWN CORRECTLY """
# utils
categories = TEST_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}
box_annotator = sv.BoxAnnotator()


# test model on test set
image_ids = TEST_DATASET.coco.getImgIds()
with sv.ImageSink(target_dir_path="./final_sets/detr_test_results", overwrite=True) as sink:
    for image_id in image_ids:
        # load image and annotatons
        image = TEST_DATASET.coco.loadImgs(image_id)[0]
        image_file_name = image['file_name']
        print('Image #{}'.format(image_id), "file name: ", image_file_name)
        image_path = os.path.join(TEST_DATASET.root, image_file_name)
        image = cv2.imread(image_path)

        with torch.no_grad():
            print("Load images and predict")
            # load image and predict
            inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
            outputs = model(**inputs)

            # post-process
            target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
            results = image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=CONFIDENCE_THRESHOLD,
                target_sizes=target_sizes
            )[0]

        # # annotate
        detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.01)
        labels = [
            f"{class_id} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        print("Labels: \n", labels)
        print("Detections: ", detections)
        frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        # sv.plot_image(image, (16, 16))
        # sv.plot_image(image, (16, 16))
        sink.save_image(image=image, image_name=image_file_name)

        with open('./final_sets/detr_test_results/'+image_file_name[:-4]+'.txt', 'w') as file:

            file.write("Image name : "+ image_file_name + "\n")
            for class_id, confidence, xyxy in zip(detections.class_id, detections.confidence, detections.xyxy):
                if class_id == 0:
                    class_name = "cystolith"
                else:
                    class_name = "fake_cystolith"
                xyxy_list = str(xyxy)[1:-1].split()
                width = float(xyxy_list[2]) - float(xyxy_list[0])
                height = float(xyxy_list[3]) - float(xyxy_list[1])
                print(xyxy_list)
                file.write(class_name + ": " + str(int(confidence*100)) +
                           "%	(left_x: " + xyxy_list[0] +
                           "   top_y: " + xyxy_list[3] +
                            "   width:  " + str(width) +
                            "   height:  " + str(height) + ")\n")
