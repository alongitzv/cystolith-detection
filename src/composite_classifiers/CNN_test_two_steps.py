
import os.path
import argparse
import yaml
import shutil
import torch.utils.data
import torchvision

from utils import *
from models import *
from data_loader import *





def test_single(img_file='', ann_file='', out_file='', model='', type=''):

    print("Start inference")

    # initiate params
    img_list = []
    do_resize = []
    do_bbx = False
    tmp_dir = ''
    if type == 'whole':
        do_resize = [214,512]
    elif type == 'yolo' or type == 'detr':
        do_resize = [256,256]
        do_bbx = True
        tmp_dir = os.path.join('runs','tmp_pics')
        os.makedirs(tmp_dir, exist_ok=True)

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
        img_list = classifier_bbx_to_images_single_pic(img_file=img_file, ann_file=ann_file, out_dir=tmp_dir, out_size=[512, 512], classifier=type)
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

 #           scores = softmax(outputs)
 #           res[i, 0] = prediction.item()
 #           res[i, 1] = scores[0][0].item()
 #           res[i, 2] = scores[0][1].item()
#
 #   np.savetxt(out_file, res, delimiter=' ',fmt='%.3f')

    print("End inference")

    return int(prediction.item())


if __name__ == "__main__":

    img_file = r"D:\Users Data\arthurSoussan\Desktop\detr\final_sets\test_2024\S(208).jpg"
    ann_file = ''
    out_file = r"D:\Users Data\arthurSoussan\Desktop\Cannabis project - sumarizing\combined_method\CNN_results.txt"
    model = r"D:\Users Data\arthurSoussan\Desktop\Cannabis project - sumarizing\combined_method\CNN_models\cfg_04_ETAZ_f1_1a\cysto_net.pth"
    type = "whole"


    # call the test routine
    res_single = test_single(img_file=img_file, ann_file=ann_file, out_file=out_file, model=model, type=type)
    print(res_single)
    print('Done')