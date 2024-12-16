

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os.path
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
from random import randrange
import math
import shutil

from data_loader import *
from models import *
from torchviz import make_dot
from torch import tensor
from torchview import draw_graph


cysto_classes = ('real', 'fake')
cysto_labels = [0,1]
IMG_SIZE = [2160, 4096]
CROPPED_H = [int((2160-1710)/2), int((2160-1710)/2 + 1710)]
CROP_H_RATIO = float(2160/1710)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def display_sample(dataset, cols=8, rows=5):
# from:
# https://medium.com/mlearning-ai/cifar10-image-classification-in-pytorch-e5185176fbef
    IDX_TO_LABEL = {v: k for k, v in dataset.class_to_idx.items()}
    fig, axs = plt.subplots(rows, cols, figsize=(12, 9))
    for x in range(rows):
        for y in range(cols):
            rnd_idx = randrange(len(dataset.data))
            axs[x, y].set_title(IDX_TO_LABEL[dataset.targets[rnd_idx]])
            axs[x, y].imshow(dataset.data[rnd_idx])
            axs[x, y].set_axis_off()
    plt.show()


def save_train_loss_info(res_dir, loss_info):
    acc_txt_name = os.path.join(res_dir,'train_acc.txt')
    acc_fig_name = os.path.join(res_dir, 'train_acc.png')
    acc_txt = np.array(loss_info)

    np.savetxt(acc_txt_name, acc_txt, fmt='%.3f', delimiter='\t')

    # colors = ['empty', 'k', 'r', 'g', 'b', 'm', 'c']
    #iters = acc_txt[:, 0]

    epochs = acc_txt[:, 0]
    iters = acc_txt[:, 1]
    val_idx = acc_txt[:, 2]
    data1 = acc_txt[:,3]
    data2 = acc_txt[:,4]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    # ax1.set_xlabel('iterations')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=color)
    # ax1.plot(iters, data1, color=color)
    ax1.plot(val_idx, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    # ax2.plot(iters, data2, color=color)
    ax2.plot(val_idx, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig(acc_fig_name)
    plt.close()


def save_test_scores(res_dir, correct_pred, total_pred):
    file_name = os.path.join(res_dir, 'test_scores.txt')
    with open(file_name, 'w') as f:
        # fmt_str = "{10:}\t{10:}\n"
        total_acc = 100 * sum(correct_pred.values()) / sum(total_pred.values())
        # print_str = fmt_str.format('total', total_acc)
        print_str = 'mean' + '\t' + '{:.2f}'.format(total_acc) + '\n'
        f.write(print_str)
        f.write('---------------------------\n')
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            # print_str = fmt_str.format(classname, accuracy)
            print_str = classname + '\t' + '{:.2f}'.format(accuracy) + '\n'
            f.write(print_str)


def save_confusion_matrix(res_dir, ground_truth_labels, predicted_labels):
    file_name = os.path.join(res_dir, 'confusion_matrix.png')
    num_labels = len(cysto_classes)
    gt_labels = [None] * num_labels
    pr_labels = [None] * num_labels
    for i in range(num_labels):
        gt_labels[i] = cysto_classes[i]   # 'GT ' + classes[i]
        pr_labels[i] = cysto_classes[i]   # 'PR ' + classes[i]

    array = confusion_matrix(ground_truth_labels, predicted_labels)
    df_cm = pd.DataFrame(array, index=[i for i in gt_labels], columns=[i for i in pr_labels])
    sn.set(font_scale=1.0)  # for label size
    ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='Blues', fmt='d')
    ax.set_xlabel('Prediction', fontsize=15)
    ax.set_ylabel('Ground Truth', fontsize=15)

    plt.savefig(file_name)
    #plt.show()
    plt.close()

    array = array.astype(float)
    array[0,:] = 100.0 * (array[0,:] / np.sum(array[0,:]))
    array[1,:] = 100.0 * (array[1,:] / np.sum(array[1,:]))
    file_name = os.path.join(res_dir, 'confusion_matrix_percent.png')
    df_cm = pd.DataFrame(array, index=[i for i in gt_labels], columns=[i for i in pr_labels])
    sn.set(font_scale=1.0)  # for label size
    ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='Blues', fmt='.1f')
    ax.set_xlabel('Prediction', fontsize=15)
    ax.set_ylabel('Ground Truth', fontsize=15)

    plt.savefig(file_name)
    plt.close()


def save_labels_predictions_filenames(res_dir, res_list):
    file_name = os.path.join(res_dir, 'test_result_list.txt')
    num_lines = len(res_list)
    f = open(file_name, 'w')
    for i in range(0,num_lines):
        f.write(res_list[i] + '\n')
    f.close()


def get_loss_function(cfg_loss):
    loss_fn = nn.CrossEntropyLoss()     # default loss (cross-entropy)
    if cfg_loss is None:
        # no loss specified, return default
        return loss_fn
    name = cfg_loss.get('name', None)
    if name is None:
        # no loss specified, return default
        return loss_fn
    elif name == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()
    return loss_fn


def get_optimizer(cfg_optimizer, net):
    optimizer = optim.Adam(net.parameters(), lr = 1e-3)    # default optimizer (Adam with learning_rate=0.001)
    if cfg_optimizer is None:
        # no optimizer specified, return default
        return optimizer
    name = cfg_optimizer.get('name', None)
    if name is None:
        # no optimizer specified, use default
        return optimizer
    elif name == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=cfg_optimizer['lr'])
    elif name == 'SGD':
        optimizer = optim.SGD(net.parameters(),
                              lr=cfg_optimizer['lr'],
                              momentum=cfg_optimizer['momentum'],
                              weight_decay = cfg_optimizer['weight_decay'])
    return optimizer


def get_scheduler(cfg_scheduler, optimizer):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)    # default scheduler ("degenerate" step LR with factor = 1.0)
    if cfg_scheduler is None:
        # no scheduler specified, return default
        return scheduler
    name = cfg_scheduler.get('name', None)
    if name is None:
        # no scheduler specified, return default
        return scheduler
    elif name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg_scheduler['step_size'], gamma=cfg_scheduler['gamma'])
    elif name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_scheduler['T_max'])
    return scheduler


def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = '1'


def create_file_lists(real_dirs, fake_dirs, config_dir, split_ratio = (0.8, 0.1, 0.1)):

    train_file_list = os.path.join(config_dir, 'train_list.txt')
    validation_file_list = os.path.join(config_dir, 'validation_list.txt')
    test_file_list = os.path.join(config_dir, 'test_list.txt')

    real_list = []
    fake_list = []
    for curr_dir in real_dirs:
        dir_list = os.listdir(curr_dir)
        for curr_file in dir_list:
            full_path = os.path.join(curr_dir,curr_file)
            real_list.append(full_path)
    for curr_dir in fake_dirs:
        dir_list = os.listdir(curr_dir)
        for curr_file in dir_list:
            full_path = os.path.join(curr_dir, curr_file)
            fake_list.append(full_path)
    num_real = len(real_list)
    num_fake = len(fake_list)
    # idx_real = np.random.permutation(num_real)
    # idx_fake = np.random.permutation(num_fake)
    random.shuffle(real_list)
    random.shuffle(fake_list)

    idx_split_real = [np.floor(split_ratio[0] * num_real).astype(int), np.floor((split_ratio[0] + split_ratio[1]) * num_real).astype(int)]
    idx_split_fake = [np.floor(split_ratio[0] * num_fake).astype(int), np.floor((split_ratio[0] + split_ratio[1]) * num_fake).astype(int)]

    fmt_str = "{:d}\t{:80}\n"

    with open(train_file_list, 'w') as f:
        for i in range(0, idx_split_real[0]):
            lbl = 0
            file_name = real_list[i]
            print_str = fmt_str.format(lbl, file_name)
            f.write(print_str)
        for i in range(0, idx_split_fake[0]):
            lbl = 1
            file_name = fake_list[i]
            print_str = fmt_str.format(lbl, file_name)
            f.write(print_str)

    with open(validation_file_list, 'w') as f:
        for i in range(idx_split_real[0], idx_split_real[1]):
            lbl = 0
            file_name = real_list[i]
            print_str = fmt_str.format(lbl, file_name)
            f.write(print_str)
        for i in range(idx_split_fake[0], idx_split_fake[1]):
            lbl = 1
            file_name = fake_list[i]
            print_str = fmt_str.format(lbl, file_name)
            f.write(print_str)

    with open(test_file_list, 'w') as f:
        for i in range(idx_split_real[1], num_real):
            lbl = 0
            file_name = real_list[i]
            print_str = fmt_str.format(lbl, file_name)
            f.write(print_str)
        for i in range(idx_split_fake[1], num_fake):
            lbl = 1
            file_name = fake_list[i]
            print_str = fmt_str.format(lbl, file_name)
            f.write(print_str)


def create_file_lists_AZ_set1(in_dir, config_dir):

    split_str = ['train', 'validation', 'test']
    img_set = str.split(in_dir, '/')[-2]  # [-1] results in empty string

    for n in range (0, len(split_str)):
        pic_dir = os.path.join(in_dir, split_str[n])
        out_file = os.path.join(config_dir, img_set + '_' + split_str[n] + '_list.txt')
        out_f = open(out_file, 'w')
        img_list = os.listdir(pic_dir)
        for i in range(0,len(img_list)):
            curr_file = img_list[i]
            # file_ext = str.split(curr_file,'.')[-1]
            # if file_ext == 'txt':
            #     continue
            # str_label = curr_file[8]
            str_label = curr_file[0]
            int_label = ''
            if str_label == 'C':
                int_label = '0'
            elif str_label == 'S':
                int_label = '1'
            else:
                print('*** invalid label ***')
            out_str = int_label + '\t' + pic_dir + os.sep + curr_file + '\n'
            out_f.write(out_str)
        out_f.close()



def change_file_list_format(in_lists, out_lists, in_prefix, out_prefix):

    fmt_str = "{:d}\t{:80}\n"

    for i in range(0,3):
        in_file = in_lists[i]
        out_file = out_lists[i]
        with open(in_file, 'r') as in_f:
            in_list = in_f.readlines()
            num_files = len(in_list)
            with open(out_file, 'w') as out_f:
                for j in range(0, num_files):
                    str = in_list[j]
                    str = str.replace('\n', '')
                    lbl = 99
                    if 'C' in str:
                        lbl = 0
                    elif 'S' in str:
                        lbl = 1
                    str = str.replace(in_prefix, out_prefix)
                    print_str = fmt_str.format(lbl, str)
                    out_f.write(print_str)



def get_mean_std_values(img_list='',img_size = [0,0]):

    data_set = cystolith_loader(split = 'train',
                                 img_list_file = img_list,
                                 img_size = img_size,
                                 do_resize = [0,0],
                                 augmentation_params = None,
                                 debug_info = None)
    # data_loader = torch.utils.data.DataLoader(data_set, batch_size = 8, shuffle = False, num_workers = 8)
    num_images = len(data_set)

    # placeholders
    psum = np.zeros(shape=(1,3), dtype=np.float64)
    psum_sq = np.zeros(shape=(1,3), dtype=np.float64)

    # loop through images
    for i in range(0, num_images):
        print(i)
        img = data_set.get_img(i)
        psum += np.sum(img, axis=(0, 1))
        psum_sq += np.sum(img ** 2,axis=(0,1))

    # pixel count
    count = num_images * img_size[0] * img_size[1]

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = np.sqrt(total_var)

    # output
    print("mean: " + str(total_mean))
    print("std:  " + str(total_std))


def adjust_bounding_box(bbx, orientation):
    new_bbx = bbx
    if orientation == 8:      # Image.ROTATE_90 (clockwise)
        new_bbx = [1.0 - bbx[1], bbx[0], bbx[3], bbx[2]]
    elif orientation == 6:    # Image.ROTATE_270
        new_bbx = [bbx[1], 1.0 - bbx[0], bbx[3], bbx[2]]
    return new_bbx


def adjust_and_crop_bounding_box(bbx, orientation):
    new_bbx = bbx
    is_valid = True
    if orientation == 8:      # Image.ROTATE_90 (clockwise)
        new_bbx = [1.0 - bbx[1], bbx[0], bbx[3], bbx[2]]
    elif orientation == 6:    # Image.ROTATE_270
        new_bbx = [bbx[1], 1.0 - bbx[0], bbx[3], bbx[2]]
    new_bbx[1] = CROP_H_RATIO * (new_bbx[1] - 0.5) + 0.5
    new_bbx[3] = CROP_H_RATIO * new_bbx[3]
    if (new_bbx[1] - 0.5 * new_bbx[3]) < 0 or (new_bbx[1] + 0.5 * new_bbx[3]) > 1:
        is_valid = False
    return new_bbx, is_valid


def adjust_image_and_annotation(in_img_file, in_ann_file, out_img_file, out_ann_file, log_file):

    str_split = in_ann_file.split(os.sep)[-2]
    img_label = str_split.split('_')[-1]
    img = Image.open(in_img_file)
    # img = ImageOps.exif_transpose(img) # DOESN'T DO THE JOB!!!
    exif = img.getexif()
    orientation = exif.get(0x0112)
    do_transpose = False
    if orientation is not None and orientation > 1:    # 1, None are "normal", 6 - rotate 270, 8 - rotate 90. No other values detected.
        do_transpose = True
    # img = np.asarray(img)
    # out_img = img
    # out_img = Image.fromarray(out_img)
    # out_img.save(out_img_file)
    img.save(out_img_file)

    in_f = open(in_ann_file, 'r')
    in_ann = in_f.readlines()
    in_f.close()

    out_f = open(out_ann_file, 'w')
    num_labels = len(in_ann)

    for i in range(0, num_labels):
        curr_line = in_ann[i]
        values = curr_line.split(' ')
        values = [float(x) for x in values]
        int_label = int(values[0])
        if (img_label == 'C' and int_label != 0) or (img_label == 'S' and int_label != 1):
            log_str = '***** wrong label *******\n'
            # print(log_str)
            log_file.write(log_str)
            # continue
        bbx = values[1:]
        new_bbx = bbx
        if do_transpose is True:
            new_bbx = adjust_bounding_box(bbx, orientation)
        new_values = str(int_label)
        for j in new_bbx:
            new_values += (' ' + str(j))
        new_values += '\n'
        out_f.write(new_values)
    out_f.close()

    log_str = in_img_file.split(os.sep)[-2] + '\t ' + in_img_file.split(os.sep)[-1] + '\t' + 'exif' + str(orientation) + '\n'
    # print(log_str)
    log_file.write(log_str)

    dummy = 0


def adjust_and_crop_image_and_annotation(in_img_file, in_ann_file, out_img_file, out_ann_file, log_file):

    str_split = in_ann_file.split(os.sep)[-2]
    img_label = str_split.split('_')[-1]
    img = Image.open(in_img_file)
    # img = ImageOps.exif_transpose(img) # DOESN'T DO THE JOB!!!
    exif = img.getexif()
    orientation = exif.get(0x0112)
    do_transpose = False
    if orientation is not None and orientation > 1:    # 1, None are "normal", 6 - rotate 270, 8 - rotate 90. No other values detected.
        do_transpose = True

    in_f = open(in_ann_file, 'r')
    in_ann = in_f.readlines()
    in_f.close()
    num_labels = len(in_ann)
    out_ann = []

    for i in range(0, num_labels):
        curr_line = in_ann[i]
        values = curr_line.split(' ')
        values = [float(x) for x in values]
        int_label = int(values[0])
        if (img_label == 'C' and int_label != 0) or (img_label == 'S' and int_label != 1):
            log_str = '***** wrong label *******\n'
            # print(log_str)
            log_file.write(log_str)
            # continue
        bbx = values[1:]
        new_bbx = bbx
        is_valid_bbx = False
        new_bbx, is_valid_bbx = adjust_and_crop_bounding_box(bbx, orientation)
        if is_valid_bbx is True:
            new_values = str(int_label)
            for j in new_bbx:
                new_values += (' ' + str(j))
            new_values += '\n'
            out_ann.append(new_values)

    log_str = in_img_file.split(os.sep)[-2] + '\t ' + in_img_file.split(os.sep)[-1] + '\t' + 'exif' + str(orientation) + '\n'
    log_file.write(log_str)

    if len(out_ann) > 0:
        img = np.asarray(img)
        img = img[CROPPED_H[0]:CROPPED_H[1], :, :]
        img = Image.fromarray(img)
        img.save(out_img_file)

        out_f = open(out_ann_file, 'w')
        for curr_line in out_ann:
            out_f.write(curr_line)
        out_f.close()
    else:
        log_str = '*** not valid after crop ***\n'
        log_file.write(log_str)
    dummy = 0


def crop_and_adjust_all(in_dir, out_dir, in_folders, do_crop = False):

    log_file_name = 'crop_and_adjust_log.txt'
    log_file = open(log_file_name, 'w')

    os.makedirs(os.path.join(out_dir, 'pics'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'anns'), exist_ok=True)

    for curr_dir in in_folders:
        print('starting  ' + curr_dir)
        curr_path = os.path.join(in_dir, 'pics', curr_dir)
        img_list = os.listdir(curr_path)
        for curr_file in img_list:
            img_file = os.path.join(curr_path,curr_file)
            ann_file = str(img_file)
            ann_file = ann_file.replace('pics', 'anns').replace('jpg', 'txt')
            if os.path.isfile(img_file) and os.path.isfile(ann_file):
                img_split = img_file.split(os.sep)
                img_set = img_split[-2]
                img_name = img_split[-1]
                ann_name = str(img_name).replace('jpg', 'txt')

                img_name = img_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                ann_name = ann_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')

                out_img_file = os.path.join(out_dir,'pics',img_set + '_' + img_name)
                out_ann_file = os.path.join(out_dir,'anns',img_set + '_' + ann_name)

                if do_crop is True:
                    adjust_and_crop_image_and_annotation(img_file, ann_file, out_img_file, out_ann_file, log_file)
                else:
                    adjust_image_and_annotation(img_file, ann_file, out_img_file, out_ann_file, log_file)

        dummy = 0
    log_file.close()
    dummy = 0


def split_train_validation_test(pic_dir, chunk_size = 100, split_ratio = [0.8, 0.1, 0.1]):
    img_dir = os.path.join(pic_dir, 'pics')
    train_dir = os.path.join(pic_dir, 'train')
    validation_dir = os.path.join(pic_dir, 'validation')
    test_dir = os.path.join(pic_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    img_list = os.listdir(img_dir)
    num_files = len(img_list)
    num_chunks = math.floor(num_files/chunk_size)
    for i in range (0,num_chunks):
        for j in range (0, int(chunk_size*split_ratio[0])):
            file_name = img_list[i * chunk_size + j]
            in_file = os.path.join(img_dir, file_name)
            out_file = os.path.join(train_dir, file_name)
            shutil.copyfile(in_file, out_file)
        for j in range (int(chunk_size * sum(split_ratio[:1])), int(chunk_size * sum(split_ratio[:2]))):
            file_name= img_list[i * chunk_size + j]
            in_file = os.path.join(img_dir, file_name)
            out_file = os.path.join(validation_dir, file_name)
            shutil.copyfile(in_file, out_file)
        for j in range(int(chunk_size *  sum(split_ratio[:2])), int(chunk_size * sum(split_ratio))):
            file_name = img_list[i * chunk_size + j]
            in_file = os.path.join(img_dir, file_name)
            out_file = os.path.join(test_dir, file_name)
            shutil.copyfile(in_file, out_file)
    # last (non-full) chunk to train folder
    for i in range(num_chunks * chunk_size, num_files):
        file_name = img_list[i]
        in_file = os.path.join(img_dir, file_name)
        out_file = os.path.join(train_dir, file_name)
        shutil.copyfile(in_file, out_file)


def copy_cropped_images(in_dir, out_dir, res_file):
    out_img_dir = os.path.join(out_dir, 'pics')
    split_dirs = ['train', 'validation', 'test']
    num_split = np.zeros((9,4), dtype=np.uint32)    # (8,3) are (pic_dirs,split_dirs + placeholders for sum of rows/vols
    for n in range(0, len(split_dirs)):
        curr_split = split_dirs[n]
        curr_out_dir = os.path.join(out_dir, curr_split)
        os.makedirs(curr_out_dir, exist_ok=True)
        curr_in_dir = os.path.join(in_dir, curr_split)
        img_list = os.listdir(curr_in_dir)
        num_files = len(img_list)
        for i in range(0, num_files):
            in_file_name = img_list[i]
            out_file_name = os.path.join(out_img_dir, in_file_name)
            if os.path.isfile(out_file_name):
                new_file_path = os.path.join(curr_out_dir,in_file_name)
                shutil.copy(out_file_name,new_file_path)
                prefix = in_file_name[5:7]
                prefix = int(prefix) - 1
                num_split[prefix, n] += 1

    cols_sum = np.sum(num_split, axis=0)
    rows_sum = np.sum(num_split, axis=1)
    num_split[-1, :] = cols_sum
    num_split[:, -1] = rows_sum
    num_split[-1, -1] = np.sum(num_split[0:8, 0:3])
    np.savetxt(res_file, num_split, fmt='%5u')
    DF = pd.DataFrame(num_split)
    DF.to_csv("data1.csv")
    dummy = 0


def copy_train_validation_test_annotation_files(pic_dir):
    ann_dir = os.path.join(pic_dir, 'anns')
    split_dirs = ['train', 'validation', 'test']
    for curr_split in split_dirs:
        curr_dir = os.path.join(pic_dir, curr_split)
        img_list = os.listdir(curr_dir)
        num_files = len(img_list)
        for i in range (0,num_files):
            curr_img_file = img_list[i]
            file_name = str.split(curr_img_file,os.sep)[-1]
            file_name = file_name.replace('jpg', 'txt')
            in_file = os.path.join(ann_dir, file_name)
            out_file = os.path.join(curr_dir, file_name)
            shutil.copyfile(in_file, out_file)


def move_bbx_from_image_border(h1, h2, w1, w2, img_size, out_size):
    if h1<0:
        h1 = 0
        h2 = out_size[0]
    if h2>img_size[0]:
        h1 = img_size[0] - out_size[0]
        h2 = img_size[0]
    if w1<0:
        w1 = 0
        w2 = out_size[1]
    if w2>img_size[1]:
        w1 = img_size[1] - out_size[1]
        w2 = img_size[1]
    return h1, h2, w1, w2


def classifier_bbx_to_images(pic_dir, ann_dir, out_dir, log_file_name, out_size, classifier):

    MOVE_BBX_FROM_EDGE = True
    KEEP_WRONG_LABEL = True
    if classifier == 'YOLO':
        BBX_START_LINE = 13
        LABEL_NAMES = ['cystolith', 'fake_cystolith']
    elif classifier == 'DETR':
        BBX_START_LINE = 1
        LABEL_NAMES = ['0', '1']
    else:
        print('*** INVALID CLASSIFIER ***')
        return

    log_file = open(log_file_name, 'w')
    in_file_list = os.listdir(ann_dir)

    num_ann_files = 0
    num_out_bbx = 0

    for ann_file in in_file_list:
        file_ext = str.split(ann_file, '.')[-1]
        if file_ext != 'txt':
            continue
        # now ann_file is the annotation file
        num_ann_files += 1
        if classifier == 'YOLO':
            # prefix = str.split(ann_file, '_')[0]
            prefix = str.split(ann_file, '_yolo_result')[0]
            img_file = os.path.join(pic_dir, prefix + '.jpg')
        elif classifier == 'DETR':
            prefix = str.split(ann_file, '.')[0]
            prefix = prefix[0] + ' ' + prefix[1:]
            # prefix = prefix[0]
            img_file = os.path.join(pic_dir, prefix + '.jpg')
        log_file.write(ann_file + '\n')
        if not os.path.isfile(img_file):
            # no corresponding image file
            log_file.write('*** no image file' + '\n')
            continue
        f = open(os.path.join(ann_dir, ann_file), 'r')
        in_lines = f.readlines()
        f.close()
        in_lines = in_lines[BBX_START_LINE:]
        num_bbx = len(in_lines)
        if num_bbx <= 0:
            # no bbx in this file
            # log_file.write('*** no bbx' + '\n')
            continue

        img_label = ''
        if 'C' in prefix and 'S' not in prefix:
            img_label = 'C'
        elif 'S' in prefix and 'C' not in prefix:
            img_label = 'S'
        if img_label != 'C' and img_label != 'S':
            # invalid image label
            log_file.write('*** invalid image label' + '\n')
            continue

        img = Image.open(img_file)
        img = np.asarray(img)
        img_size = img.shape

        for i in range (0, num_bbx):
            curr_line = in_lines[i]
            if classifier == 'YOLO':
                tokens = str.split(curr_line, ':')
                bbx_label = ''
                if tokens[0] == LABEL_NAMES[0]:
                    bbx_label = 'C'
                elif tokens[0] == LABEL_NAMES[1]:
                    bbx_label = 'S'
                if bbx_label != img_label:
                    log_file.write('*** invalid bbx label - line ' + str(i + 1) + '\n')
                    if KEEP_WRONG_LABEL is False:
                        continue
                curr_line = curr_line.replace(')','').replace('(','').replace('-',' ')
                bbx_values = [int(s) for s in str.split(curr_line) if s.isdigit()]

                h1 = round(bbx_values[1] + 0.5 * (bbx_values[3] - out_size[0]))
                h2 = h1 + out_size[0]
                w1 = round(bbx_values[0] + 0.5 * (bbx_values[2] - out_size[1]))
                w2 = w1 + out_size[1]

            elif classifier == 'DETR':
                tokens = str.split(curr_line, ',')
                token = tokens[0][-1]
                bbx_label = ''
                if token == LABEL_NAMES[0]:
                    bbx_label = 'C'
                elif token == LABEL_NAMES[1]:
                    bbx_label = 'S'
                if bbx_label != img_label:
                    log_file.write('*** invalid bbx label - line ' + str(i + 1) + '\n')
                    if KEEP_WRONG_LABEL is False:
                        continue

                # curr_values = str.split(curr_line, '[')[-1]
                # curr_values = str.split(curr_values, ']')[0]
                # bbx_values = [round(float(s)) for s in str.split(curr_values)]
                # h1 = round(bbx_values[1] + 0.5 * (bbx_values[3] - bbx_values[1] - out_size[0]))
                # h2 = h1 + out_size[0]
                # w1 = round(bbx_values[0] + 0.5 * (bbx_values[2] - bbx_values[0] - out_size[1]))
                # w2 = w1 + out_size[1]

                curr_values = str.split(curr_line, '(')[-1]
                curr_values = str.split(curr_values, ')')[0]
                curr_values = str.split(curr_values, ' ')
                curr_values = [curr_values[1], curr_values[5], curr_values[10], curr_values[15]]
                bbx_values = [round(float(s)) for s in curr_values]
                h2 = round(bbx_values[1] - 0.5 * (bbx_values[3] - out_size[0]))
                h1 = h2 - out_size[0]
                w1 = round(bbx_values[0] + 0.5 * (bbx_values[2] - out_size[1]))
                w2 = w1 + out_size[1]

            if h1 < 0 or h2 > img_size[0] or w1 < 0 or w2 > img_size[1]:
                if MOVE_BBX_FROM_EDGE is True:
                    h1, h2, w1, w2 = move_bbx_from_image_border(h1, h2, w1, w2, img_size, out_size)
                    log_file.write('*** moving bbx from edge - line ' + str(i + 1) + '\n')

            if h1<0 or h2>img_size[0] or w1<0 or w2>img_size[1]:
                log_file.write('*** invalid bbx area - line ' + str(i + 1) + '\n')
                continue

            out_img = img[h1:h2, w1:w2, :]
            out_img = Image.fromarray(out_img)
            out_name = prefix.replace(' ','_').replace('(','').replace(')','')
            if classifier == 'YOLO':
                out_name = out_name + '_y_'
            elif classifier == 'DETR':
                out_name = out_name + '_d_'
            out_name = out_name + str(i + 1).zfill(3) + '.png'
            out_img_file = os.path.join(out_dir, out_name)
            out_img.save(out_img_file)
            num_out_bbx += 1

    log_file.write('\n')
    log_file.write('num_ann_files: ' + str(num_ann_files) + '\n')
    log_file.write('num_out_bbx: ' + str(num_out_bbx) + '\n')
    log_file.close()
    dummy = 0


def classifier_bbx_to_images_single_pic(img_file, ann_file, out_dir, out_size, classifier):

    MOVE_BBX_FROM_EDGE = True
    if classifier == 'yolo':
        BBX_START_LINE = 13
    elif classifier == 'detr':
        BBX_START_LINE = 1
    else:
        print('*** INVALID CLASSIFIER TYPE ***')
        return

    img_list = []
    prefix = str.split(img_file, '/')[-1]
    prefix = str.split(prefix, '.')[0]

    f = open(ann_file, 'r')
    in_lines = f.readlines()
    f.close()
    in_lines = in_lines[BBX_START_LINE:]
    num_bbx = len(in_lines)
    if num_bbx <= 0:
        # no bbx in this file
        return img_list

    img = Image.open(img_file)
    img = np.asarray(img)
    img_size = img.shape

    for i in range (0, num_bbx):
        curr_line = in_lines[i]
        if classifier == 'yolo':
            curr_line = curr_line.replace(')','').replace('(','').replace('-',' ')
            bbx_values = [int(s) for s in str.split(curr_line) if s.isdigit()]
            h1 = round(bbx_values[1] + 0.5 * (bbx_values[3] - out_size[0]))
            h2 = h1 + out_size[0]
            w1 = round(bbx_values[0] + 0.5 * (bbx_values[2] - out_size[1]))
            w2 = w1 + out_size[1]
        elif classifier == 'detr':
            # curr_values = str.split(curr_line, '[')[-1]
            # curr_values = str.split(curr_values, ']')[0]
            # bbx_values = [round(float(s)) for s in str.split(curr_values)]
            # h1 = round(bbx_values[1] + 0.5 * (bbx_values[3] - bbx_values[1] - out_size[0]))
            # h2 = h1 + out_size[0]
            # w1 = round(bbx_values[0] + 0.5 * (bbx_values[2] - bbx_values[0] - out_size[1]))
            # w2 = w1 + out_size[1]

            curr_values = str.split(curr_line, '(')[-1]
            curr_values = str.split(curr_values, ')')[0]
            curr_values = str.split(curr_values, ' ')
            curr_values = [curr_values[1], curr_values[5], curr_values[10], curr_values[15]]
            bbx_values = [round(float(s)) for s in curr_values]
            h2 = round(bbx_values[1] - 0.5 * (bbx_values[3] - out_size[0]))
            h1 = h2 - out_size[0]
            w1 = round(bbx_values[0] + 0.5 * (bbx_values[2] - out_size[1]))
            w2 = w1 + out_size[1]

        if h1 < 0 or h2 > img_size[0] or w1 < 0 or w2 > img_size[1]:
            if MOVE_BBX_FROM_EDGE is True:
                h1, h2, w1, w2 = move_bbx_from_image_border(h1, h2, w1, w2, img_size, out_size)
                print('*** moving bbx from edge - line ' + str(i + 1) + '\n')
        if h1<0 or h2>img_size[0] or w1<0 or w2>img_size[1]:
            print('*** invalid bbx area - line ' + str(i + 1) + '\n')
            continue

        out_img = img[h1:h2, w1:w2, :]
        out_img = Image.fromarray(out_img)
        out_name = prefix + '_' + str(i+1).zfill(3) + '.png'
        out_img_file = os.path.join(out_dir, out_name)
        out_img.save(out_img_file)
        img_list.append(out_img_file)

    return img_list


def create_yolo_detr_split_lists_WALK_LISTS(in_split_files, out_split_files, pic_dir):

    img_list = os.listdir(pic_dir)
    num_out = np.zeros((3), dtype=np.uint32)

    for n in range(0,3):
        in_file_name = in_split_files[n]
        in_file = open(in_file_name, 'r')
        in_lines = in_file.readlines()
        in_file.close()
        out_file_name = out_split_files[n]
        out_file = open(out_file_name, 'w')
        num_lines = len(in_lines)

        for i in range(0, num_lines):
            curr_line = in_lines[i]
            prefix = os.path.split(curr_line)[-1]
            prefix = str.split(prefix, '.')[0]
            prefix = prefix.replace(' ', '_').replace('(','').replace(')','')
            lbl = ''
            if 'C' in prefix:
                lbl = '0'
            elif 'S' in prefix:
                lbl = '1'
            prefix = prefix + '_'
            bbx_names = [name for name in img_list if prefix in name]
            for img_name in bbx_names:
                print(i)
                print(prefix)
                print(curr_line)
                out_line = lbl + '\t' + os.path.join(pic_dir, img_name) + '\n'
                out_file.write(out_line)
                num_out[n] = num_out[n] + 1
        out_file.close()
        dummy = 0
    dummy = 0
    print(num_out)


def create_yolo_detr_split_lists_WALK_IMAGES(in_split_files, out_split_files, pic_dir):

    img_list = os.listdir(pic_dir)
    num_files = len(img_list)
    num_out = np.zeros((3), dtype=np.uint32)

    for n in range(0,3):
        in_file_name = in_split_files[n]
        in_file = open(in_file_name, 'r')
        in_lines = in_file.readlines()
        in_file.close()
        out_file_name = out_split_files[n]
        out_file = open(out_file_name, 'w')
        num_lines = len(in_lines)

        for i in range(0, num_files):
            img_file_name = img_list[i]
            prefix = str.split(img_file_name, '.')[0]
            prefix = prefix[:len(prefix)-6]
            lbl = ''
            if 'C' in prefix:
                lbl = '0'
            elif 'S' in prefix:
                lbl = '1'
            orig_name = prefix
            if orig_name[0] == 'C' or orig_name[0] == 'S':
                orig_name = orig_name.replace('_',' (') + ')'
            for j in range(0,num_lines):
                curr_line = in_lines[j]
                curr_name = os.path.split(curr_line)[-1]
                curr_name = str.split(curr_name, '.')[0]
                if orig_name == curr_name:
                    out_line = lbl + '\t' + os.path.join(pic_dir, img_file_name) + '\n'
                    out_file.write(out_line)
                    num_out[n] = num_out[n] + 1
                    continue
        out_file.close()
        dummy = 0
    dummy = 0
    print(num_out)


def relabel_S_anns(in_dir, out_dir):
    all_files = os.listdir(in_dir)
    num_files = len(all_files)
    for i in range(0, num_files):
        in_file_name = os.path.join(in_dir, all_files[i])
        out_file_name = os.path.join(out_dir, all_files[i])
        in_file = open(in_file_name, 'r')
        in_lines = in_file.readlines()
        in_file.close()
        num_lines = len(in_lines)
        out_file = open(out_file_name, 'w')
        for j in range(0,num_lines):
            if j>0:
                prev_line = str(curr_line)
            curr_line = in_lines[j]
            curr_line = curr_line[:0] + '1' + curr_line[1:]
            if j==0:
                out_file.write(curr_line)
            elif prev_line != curr_line:
                 out_file.write(curr_line)
        out_file.close()


def count_num_annotations(ann_dir, pic_dir):
    all_files = os.listdir(ann_dir)
    num_files = len(all_files)
    num_bbx = np.zeros(2, dtype=np.uint32)
    min_size = np.inf
    max_size = 0
    total_wh_real = np.zeros(2, dtype=np.long)
    total_wh_fake = np.zeros(2, dtype=np.long)

    for i in range(0, num_files):
        in_file_name = os.path.join(ann_dir, all_files[i])

        prefix = str.split(all_files[i], '.')
        in_pic_name = prefix[0] + '.jpg'
        in_pic_name = os.path.join(pic_dir, in_pic_name)
        if not os.path.exists(in_pic_name):
            print(all_files[i])
            continue

        in_file = open(in_file_name, 'r')
        in_lines = in_file.readlines()
        in_file.close()
        num_lines = len(in_lines)
        for j in range(0, num_lines):
            curr_line = in_lines[j]
            vals = str.split(curr_line, ' ')
            lbl = int(vals[0])
            w = int(float(vals[3]) * 4096)
            h = int(float(vals[4]) * 1710)
            if w<10 or h<10:
                print('*** ' + str(w) + ' ' + str(h) + ' ' + all_files[i])
                dummy = 0
                continue
            curr_size = w * h
            if curr_size < min_size:
                min_size = curr_size
                min_bbx = [w, h]
            if curr_size > max_size:
                max_size = curr_size
                max_bbx = [w, h]
            num_bbx[lbl] += 1
            if lbl == 0:
                total_wh_real[0] += w
                total_wh_real[1] += h
            else:
                total_wh_fake[0] += w
                total_wh_fake[1] += h

    avg_real = total_wh_real / float(num_bbx[0])
    avg_fake = total_wh_fake / float(num_bbx[1])

    print(num_bbx)
    print(min_size)
    print(min_bbx)
    print(max_size)
    print(max_bbx)
    print('avg :')
    print(avg_real)
    print(avg_fake)
    dummy = 0


if __name__ == "__main__":

    ##############################################################################
    # create file lists

    # data_dir_1 = 'C:/alon/datasets/cystolith/set1/'
    # data_dir_2 = 'C:/alon/datasets/cystolith/set2/'
    #
    # real_dirs = [os.path.join(data_dir_1,'real'),
    #              os.path.join(data_dir_2,'real'),
    #              ]
    #
    # fake_dirs = [os.path.join(data_dir_1,'fake'),
    #              os.path.join(data_dir_2,'fake'),
    #              ]
    #
    # split_ratio = (0.8, 0.1, 0.1)     # train / validation / test
    # config_dir = 'C:/alon/cystolith/configs/'
    #
    # create_file_lists(real_dirs, fake_dirs, config_dir, split_ratio)

    # in_dir = 'C:/alon/datasets/cystolith/AZ_set1_cropped/'
    # in_dir = 'C:/alon/datasets/cystolith/AZ_set1_no_crop/'
    # in_dir = 'C:/alon/datasets/cystolith/AZ_set2_cropped/'
    # in_dir = 'C:/alon/datasets/cystolith/ET_f2/'
    # config_dir = 'C:/alon/cystolith/configs/'

    # create_file_lists_AZ_set1(in_dir, config_dir)

    ##############################################################################
    # get mean + std values

    # img_list = 'configs/all_images.txt'
    # img_size = [1710, 4096]     # [2160, 4096]
    # get_mean_std_values(img_list, img_size)

    ##############################################################################
    # change Emma & Tamar's lists to my format

    # config_dir = 'configs/'
    # in_lists = [os.path.join(config_dir, 'ET_lists_1/train.txt'),
    #             os.path.join(config_dir, 'ET_lists_1/val.txt'),
    #             os.path.join(config_dir, 'ET_lists_1/test.txt')]
    # out_lists = [os.path.join(config_dir, 'train_list.txt'),
    #              os.path.join(config_dir, 'validation_list.txt'),
    #              os.path.join(config_dir, 'test_list.txt')]
    # in_prefix = 'data/obj/'
    # out_prefix = 'C:/alon/datasets/cystolith/ET_set1' + os.sep
    # change_file_list_format(in_lists, out_lists, in_prefix, out_prefix)

    ##############################################################################

    # rename files to setXXX_L_img_YYY , where XXX,YYY are numbers, and L is class label (C or S)
    # crop all images to 4096 x 1710
    # rotate 90 degrees if needed
    # adjust bounding box coordinates to cropped images

    # in_dir = 'C:/alon/datasets/cystolith/amizur_orig/'
    # # out_dir = 'C:/alon/datasets/cystolith/AZ_set1_no_crop/'
    # # do_crop = False
    # out_dir = 'C:/alon/datasets/cystolith/AZ_set1_cropped/'
    # do_crop = True
    #
    # in_folders = ['pics_01_C',
    #               'pics_02_S',
    #               'pics_03_C',
    #               'pics_04_S',
    #               'pics_05_S',
    #               'pics_06_C',
    #               'pics_07_S',
    #               'pics_08_S']

    # in_dir = 'C:/alon/datasets/cystolith/zz_tst_in/'
    # out_dir = 'C:/alon/datasets/cystolith/zz_tst_out_cropped/'  # zz_tst_out_cropped
    # do_crop = True # True  False
    # in_folders = ['pics_42_S']

    # crop_and_adjust_all(in_dir, out_dir, in_folders, do_crop)

    ##############################################################################

    # split image folder to train/validation/test
    # copies the originals to separate directories
    # note - still need to manually separate at the plant (not image) level
    # note - need to copy the corresponding annotation files

    # pic_dir = 'C:/alon/datasets/cystolith/AZ_set1_cropped/'
    # pic_dir = 'C:/alon/datasets/cystolith/AZ_set2_cropped/'
    # split_train_validation_test(pic_dir, chunk_size=100, split_ratio=[0.8, 0.1, 0.1])
    # now manually separate at the plant (not image) level

    # copy_train_validation_test_annotation_files(pic_dir)

    ##############################################################################

    # copy cropped image files to train/validation/test directories

    # in_dir = 'C:/alon/datasets/cystolith/AZ_set1_no_crop/'
    # out_dir = 'C:/alon/datasets/cystolith/AZ_set1_cropped/'
    # res_file = 'num_cropped_files.txt'
    # in_dir = 'C:/alon/datasets/cystolith/AZ_set2_no_crop/'
    # out_dir = 'C:/alon/datasets/cystolith/AZ_set2_cropped/'
    # res_file = 'num_cropped_files_set2.txt'

    # copy_cropped_images(in_dir, out_dir, res_file)

    ##############################################################################

    # extract labeled bounding boxes from YOLO/DETR as images for 2nd stage classifier

    # pic_dir = 'C:/alon/datasets/cystolith/ET_set1_all'
    # ann_dir = 'C:/alon/datasets/cystolith/yolo_detr_results_01/yolo_2023_01/pics_anns/'
    # out_dir = 'C:/alon/datasets/cystolith/yolo_detr_bbx_images_01/yolo_2023_01/'  # 256  512
    # log_file = 'bbx_to_images_log_yolo_2023_01.txt'  # 256  512
    # classifier = 'YOLO'


    # # pic_dir = 'C:/alon/datasets/cystolith/ET_set1/'
    # pic_dir = 'C:/alon/datasets/cystolith/AZ_set1_cropped/pics/'
    # ann_dir = 'C:/alon/datasets/cystolith/yolo_detr_results_01/yolo_2024_f1/'
    # out_dir = 'C:/alon/datasets/cystolith/yolo_detr_bbx_images_01/yolo_2024_f1/'  # 256  512
    # log_file = 'bbx_to_images_yolo_2024_f1.txt'  # 256  512
    # classifier = 'YOLO'

    # # pic_dir = 'C:/alon/datasets/cystolith/ET_set1/'
    # pic_dir = 'C:/alon/datasets/cystolith/AZ_set1_cropped/pics/'
    # ann_dir = 'C:/alon/datasets/cystolith/yolo_detr_results_01/yolo_2024_f2/'
    # out_dir = 'C:/alon/datasets/cystolith/yolo_detr_bbx_images_01/yolo_2024_f2/'  # 256  512
    # log_file = 'bbx_to_images_yolo_2024_f2.txt'  # 256  512
    # classifier = 'YOLO'

    # pic_dir = 'C:/alon/datasets/cystolith/ET_set1/'
    # # pic_dir = 'C:/alon/datasets/cystolith/AZ_set1_cropped/pics/'
    # ann_dir = 'C:/alon/datasets/cystolith/yolo_detr_results_01/detr_2024_f1/'
    # out_dir = 'C:/alon/datasets/cystolith/yolo_detr_bbx_images_01/detr_2024_f1/'  # 256  512
    # log_file = 'bbx_to_images_detr_2024_f1.txt'  # 256  512
    # classifier = 'DETR'

    # # pic_dir = 'C:/alon/datasets/cystolith/ET_set1/'
    # pic_dir = 'C:/alon/datasets/cystolith/AZ_set1_cropped/pics/'
    # ann_dir = 'C:/alon/datasets/cystolith/yolo_detr_results_01/detr_2024_f2/'
    # out_dir = 'C:/alon/datasets/cystolith/yolo_detr_bbx_images_01/detr_2024_f2/'  # 256  512
    # log_file = 'bbx_to_images_detr_2024_f2.txt'  # 256  512
    # classifier = 'DETR'

    # out_size = [512, 512]   # [256, 256] [512, 512]
    # classifier_bbx_to_images(pic_dir, ann_dir, out_dir, log_file, out_size, classifier)

    ##############################################################################

    # use 2023 train/val/test split to create corresponding split from yolo/detr bbx images

    # pic_dir = 'C:/alon/datasets/cystolith/yolo_detr_bbx_images_01/yolo_2023_01'
    # in_split_files = ['C:/alon/cystolith/configs/ET_lists_1/train.txt',
    #                   'C:/alon/cystolith/configs/ET_lists_1/val.txt',
    #                   'C:/alon/cystolith/configs/ET_lists_1/test.txt']
    # out_split_files = ['C:/alon/cystolith/configs/yolo_bbx_2023_01_train.txt',
    #                    'C:/alon/cystolith/configs/yolo_bbx_2023_01_validation.txt',
    #                    'C:/alon/cystolith/configs/yolo_bbx_2023_01_test.txt']
    # in_split_files = ['C:/alon/cystolith/configs/image_lists/ET_2023_f2_train_list.txt',
    #                   'C:/alon/cystolith/configs/image_lists/ET_2023_f2_validation_list.txt',
    #                   'C:/alon/cystolith/configs/image_lists/ET_2023_f2_test_list.txt']
    # out_split_files = ['C:/alon/cystolith/configs/image_lists/yolo_bbx_2023_02_train.txt',
    #                    'C:/alon/cystolith/configs/image_lists/yolo_bbx_2023_02_validation.txt',
    #                    'C:/alon/cystolith/configs/image_lists/yolo_bbx_2023_02_test.txt']

    # pic_dir = 'C:/alon/datasets/cystolith/yolo_detr_bbx_images_01/bbx_f1/'
    # in_split_files = ['C:/alon/cystolith/configs/image_lists/ETAZ_f1_train_list.txt',
    #                   'C:/alon/cystolith/configs/image_lists/ETAZ_f1_validation_list.txt',
    #                   'C:/alon/cystolith/configs/image_lists/ETAZ_f1_test_list.txt']
    # out_split_files = ['C:/alon/cystolith/configs/image_lists/bbx_f1_train.txt',
    #                    'C:/alon/cystolith/configs/image_lists/bbx_f1_validation.txt',
    #                    'C:/alon/cystolith/configs/image_lists/bbx_f1_test.txt']

    # pic_dir = 'C:/alon/datasets/cystolith/yolo_detr_bbx_images_01/bbx_f2/'
    # in_split_files = ['C:/alon/cystolith/configs/image_lists/ETAZ_f2_train_list.txt',
    #                   'C:/alon/cystolith/configs/image_lists/ETAZ_f2_validation_list.txt',
    #                   'C:/alon/cystolith/configs/image_lists/ETAZ_f2_test_list.txt']
    # out_split_files = ['C:/alon/cystolith/configs/image_lists/bbx_f2_train.txt',
    #                    'C:/alon/cystolith/configs/image_lists/bbx_f2_validation.txt',
    #                    'C:/alon/cystolith/configs/image_lists/bbx_f2_test.txt']

    # create_yolo_detr_split_lists_WALK_LISTS(in_split_files, out_split_files, pic_dir)
    # create_yolo_detr_split_lists_WALK_IMAGES(in_split_files, out_split_files, pic_dir)

    ##############################################################################

    # change label '0' to '1' in synthetic-cannabis images
    # in_dir = 'D:/alon/zz_tmp_1/'
    # out_dir = 'D:/alon/zz_tmp_2/'
    # relabel_S_anns(in_dir, out_dir)

    ##############################################################################

    # ann_dir = 'D:/alon/cystolith_detection_for_export/cystolith_detection/cystolith-detection/data/anns/'
    # pic_dir = 'D:/alon/cystolith_detection_for_export/cystolith_detection/cystolith-detection/data/pics/'
    # count_num_annotations(ann_dir, pic_dir)

    ##############################################################################


    dummy = 0