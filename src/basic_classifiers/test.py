
import os.path
import argparse
import yaml
import shutil
import torch.utils.data
import torchvision

from utils import *
from models import *
from data_loader import *


def test(cfg):

    print("Start test")

    # make sure out_dir exists
    out_dir = cfg['train']['out_dir']
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # create test dataset and dataloader
    data_dir = cfg['data']['path']
    test_set = cystolith_loader(split = 'test',
                                img_list_file = cfg['data']['test_list'],
                                img_size = cfg['data']['img_size'],
                                do_resize = cfg['data']['do_resize'],
                                img_norm = cfg['data'].get('img_norm',False))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg['data']['n_workers'])

    # specify device, CUDA if available, CPU otherwise
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # create a neural network and load the specified model
    net = get_net(cfg)
    net.to(device)
    model_path = os.path.join(cfg['train']['out_dir'],cfg['train']['saved_model'])
    model_exists = os.path.exists(model_path)
    if model_exists:
        state = torch.load(model_path)
        net.load_state_dict(state['model_state'])
    else:
        print('Exit - no model found')
        exit()
    net.eval()

    # prepare to count predictions for each class
    correct_pred = {class_name: 0 for class_name in cysto_classes}
    total_pred = {class_name: 0 for class_name in cysto_classes}
    gt_labels = []
    predicted_labels = []

    save_test_info =  cfg['debug_info'].get('save_test_info', 0)
    if save_test_info == 1:
        res_list = []

    # no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels, file_names = data
            images, labels = images.to(device), labels.to(device)
            gt_labels.extend(labels.tolist())
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            predicted_labels.extend(predictions.tolist())
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[cysto_classes[label]] += 1
                total_pred[cysto_classes[label]] += 1

            if save_test_info == 1:
                # file_list.append(file_names)
                # all_labels.append(labels)
                # all_predictions.append(predictions)
                for label, prediction, file_name in zip(labels, predictions, file_names):
                    res_str = '     '
                    if label.item() != prediction.item():
                        res_str = '***  '
                    res_str = res_str + str(label.item()) + '  ' + str(prediction.item()) + '  ' + file_name
                    # print(res_str)
                    res_list.append(res_str)

    # print mean accuracy
    mean_acc = 100 * sum(correct_pred.values()) / sum(total_pred.values())
    print(f'Mean accuracy of the network on the {len(test_set)} test images: {mean_acc:.2f} %')

    # print accuracy for each class
    for class_name, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[class_name]
        print(f'Accuracy for class: {class_name:5s} is {accuracy:.2f} %')

    # save accuracy scores to a text file
    save_test_scores(out_dir, correct_pred, total_pred)

    # save image of confusion matrix
    save_confusion_matrix(out_dir, gt_labels, predicted_labels)

    if save_test_info == 1:
        # save list of labels, predictions, file_names
        save_labels_predictions_filenames(out_dir, res_list)

    print("End test")


if __name__ == "__main__":

    # parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/blabla.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    # check that config file exists
    if not os.path.isfile(args.config):
        print('Configuration file not found')
        exit()

    # load configuration
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    cfg['config_file'] = args.config

    # call the test routine
    test(cfg)