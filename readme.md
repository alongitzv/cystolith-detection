
---

### **Cystolith Detection**

# Cystolith Detection: A Vision-Based Deep Learning Framework

## Table of Contents
1. Introduction
2. Project Structure
3. Basic Classifiers: Overview and Usage
4. YOLO Object Detection: Training and Testing
5. DETR Object Detection: Training and Testing
6. Composite Methods
7. General Notes
8. Credits


## 1. Introduction

This repository hosts the source code and data for detecting genuine cannabis from non-cannabis plant materials using deep learning and computer vision techniques. 
Designed to aid forensic laboratories and law enforcement agencies, this project leverages advanced image analysis to distinguish genuine cannabis plants from illicit substitutes, such as synthetic cannabinoid-sprayed materials, with an accuracy exceeding 97%.

The project incorporates a pipeline of classifiers, object detectors (YOLO, DETR), and composite methods trained on thousands of annotated microscope images. 
It provides a scalable, cost-effective alternative to conventional tests, significantly reducing the time and resources required for forensic identification. 
All tools and configurations are made publicly available to ensure reproducibility and enable further research.

![Sample trichomes](images/Fig_2_trichome_samples.jpg)
Samples of non-glandular trichome hairs in cannabis (left) and non-cannabis trichomes (right) from the dataset collected during this research.

Source code and configuration files are in the 'src' folder. 
The code contains :
	1. two different classifiers for whole images.
	2. YOLO and DETR object detectors for trichome hair identification.
	3. two composite methods for classification, combining the object detectors and whole image and/or bounding box classification.

Images and annotations are in the 'data' folder.
Manual annotations of bonding boxes surrounding trichome hairs were obtained using MakeSense, freely available at https://www.makesense.ai/.
Lists of images are in the  'src/image_lists' folder.
These contain two different splits of the images into train/validation/test partitions.

For usage, first clone this repository into your local 'cystolith_detection' directory.


## 2. Project Structure

* `src/`: Contains source code for classifiers, object detection models, and composite methods.
* `data/`: Includes images and annotations used for training and testing.
* `src/image_lists/`: Contains train/validation/test partitions for the datasets.
* `src/configs/`: Configuration files for training and testing.
*  Notes on file names and annotations:
    all filenames containing 'C' refer to genuine cannabis; filenames containing 'S' refer to non-cannabis material.
    annotated bounding boxes of genuine cannabis are labeled '0'; non-cannabis are labeled '1'. 


## 3. Basic Classifiers: Overview and Usage

* Basic CNN (Convolutional Neural Network) is based on:
	https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
	https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch

* DLA (Deep Layer Aggregation) is based on:
	https://github.com/kuangliu/pytorch-cifar/blob/master/models/dla.py
	https://arxiv.org/abs/1707.06484


* Use the configuration files in \cystolith_detection\src\basic_classifiers\configs
* Use the image lists in \cystolith_detection\src\basic_classifiers\image_lists
* In the configuration file:
	1. change the data->path field to your local directory containing the images
	2. change the train->out_dir field to your desired local directory for writing results

* Small CNN training and testing:	
	python train.py --config configs/config_03_ETAZ_f1_1b4.yml
	python test.py --config configs/config_03_ETAZ_f1_1b4.yml

* DLA training and testing:
	python train.py --config configs/config_04_ETAZ_f1_1a.yml
	python test.py --config configs/config_04_ETAZ_f1_1a.yml


## 4. YOLO Object Detection: Training and Testing

* Installation and usage based on:
	https://medium.com/p/61a659d4868#e5b4
	https://github.com/AlexeyAB/darknet
	https://arxiv.org/abs/2004.10934
	
* YOLO training:

	1. open file \cystolith_detection\src\yolov4\darknet\cfg\yolov4-custom.cfg
	2. comment the lines "batch=1" and "subdivisions=1", and uncomment lines"#batch=64" and "#subdivisions=16" (lines 3,4 and 6,7)
	3. open cmd from \cystolith_detection\src\yolov4\darknet
	4. run command: darknet.exe detector train data/obj.data cfg/yolov4-custom.cfg yolov4/training/yolov4-custom_last.weights -map

* YOLO testing:

	1. open file \cystolith_detection\src\yolov4\darknet\cfg\yolov4-custom.cfg
	2. uncomment the lines "#batch=1" and "#subdivisions=1", and comment lines"batch=64" and "subdivisions=16" (lines 3,4 and 6,7)
	3. open cmd from \cystolith_detection\src\yolov4\darknet
	4. run command: py yolo_accuracy_check.py


## 5. DETR Object Detection: Training and Testing

* Installation and usage based on:
	https://github.com/roboflow/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb
	https://www.youtube.com/watch?v=AM8D4j9KoaU&t=619s
	https://link.springer.com/chapter/10.1007/978-3-030-58452-8_13

* DETR training + testing:

	1. open cmd from \cystolith_detection\src\detr\model_DETR\detr_training_and_testing
	2. run command: py detr_training.py

* DETR testing only:

	1. open cmd from \cystolith_detection\src\detr\model_DETR\detr_testing_and_analyzing
	2. run command: py detr_testing.py

* DETR analyzing:

	1. open cmd from \cystolith_detection\src\detr\model_DETR\detr_testing_and_analyzing
	2. run command: py detr_accuray_check.py


## 6. Composite Methods

* Create the bounding box dataset needed for training the composite 2&3 stage method:
	1. train DETR with a lower threshold (see previous section) and use TH = XXX in the XXX file. (TODO - find confidence_threshold, detection_threshold in detr_predict())
	2. run the classifier_bbx_to_images() function in src/basic_classifiers/utils.py to produce the bounding boxes images from DETR's classification.
	3. run the create_yolo_detr_split_lists() function in src/basic_classifiers/utils.py to create lists of train/validation/test splits of the bounding box images.
	4. train the DLA model on boundig boxes images, with the following command (similar to DLA training as described above): 
		python train.py --config configs/config_06_f1_1a_g7.yml	
	5. use the model trained on bounding boxes as part of the composite method described below.

* Composite method 1 - DETR --> CNN on whole image:

	1. open cmd from \cystolith_detection\src\composite_classifiers
	2. run command: py final_method_whole.py
	3. the output table with all detection info and predictions will be in folder \cystolith_detection\src\composite_classifiers\final_method_results

* Composite method 2 - DETR --> CNN on bounding boxes:

	1. open cmd from \cystolith_detection\src\composite_classifiers
	2. run command: py final_method_bb.py
	3. the output table with all detection info and predictions will be in folder \cystolith_detection\src\composite_classifiers\final_method_results


## 7. General Notes

* Update file paths to your local directories as required by individual scripts.
* Ensure all dependencies are installed before running any script.


## 8. Credits

"Identification of Non-Glandular Trichome Hairs in Cannabis using Vision-Based Deep Learning Methods"
Alon Zvirin ^1, Amitzur Shapira ^2, Emma Attal ^1, Tamar Gozlan ^1, Arthur Soussan ^1, Dafna De La Vega ^2, Yehudit Harush ^2, and Ron Kimmel ^1.
    1 Computer Science Department, Technion - Israel Institute of Technology. Haifa, Israel. 
    2 The Division of Forensic Sciences, National Police Headquarters, Jerusalem, Israel.
The paper will soon be published.
Contact - alongitzv@gmail.com


## END ##
