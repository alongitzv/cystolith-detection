import os
from os import getcwd
import subprocess
import time
import shutil

wd = getcwd()

image_paths_file = "data/test.txt"
obj_dat = "data\\obj.data"
yolo_cnf = "cfg\\yolov4-custom.cfg"
yolo_wghts = "..\\training\\yolov4-custom_best.weights"
path_to_ground_truth = "data\\obj"

im_paths = open(image_paths_file).read().strip().split('\n')

cfgfilename = yolo_cnf.split('\\')[1].split('.')[0]
print(cfgfilename)
path_to_save_predictions = "predictions-%s/"%(cfgfilename)

if not os.path.exists(path_to_save_predictions):
    os.makedirs(path_to_save_predictions)

for im in im_paths:
    if len(im) == 0:
        continue
    cmd = ".\\darknet detector test %s %s %s -dont_show %s"%(obj_dat, yolo_cnf, yolo_wghts, '"'+im+'"')
    print('cmd = %s'%(cmd))
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    # wait max 6s
    p.wait(6)
    # make sure child process exit normally
    if p.poll() != 0:
        print("picture %s predict fails\n"%(im))
        break
    image_name = (im.split('/')[-1]).split('.')[0]
    os.rename("predictions.jpg", os.path.join(path_to_save_predictions, image_name+'.jpg'))
    # cp groud truth near the predictions dir
    # already draw box and saved in path_to_ground_truth dir, if not want to copy, just comment the next line
    shutil.copy(path_to_ground_truth+'/'+image_name+'.jpg', path_to_save_predictions+'/'+image_name+"-truth.jpg")
    print(image_name)