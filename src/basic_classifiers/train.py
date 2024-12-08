
import os.path
import argparse
import yaml
import shutil
import torch.utils.data
import torchvision

from utils import *
from models import *
from data_loader import *


def validate(net, validation_loader, device):

    correct = 0
    total = 0

    # switch to evaluation mode
    net.eval()

    with torch.no_grad():
        for data in validation_loader:
            # calculate outputs by running images through the network
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    #print(f'Accuracy of the network = {100 * correct // total} %')
    return 100 * correct / total


def train(cfg):

    print("Start train")

    print("CUDA version:  " + torch.version.cuda)

    # random seed for reproducibility
    rand_seed = cfg.get('random_seed', None)
    if rand_seed is None:
        rand_seed = 2024
    seed_everything(seed=rand_seed)

    # make sure out_dir exists
    out_dir = cfg['train']['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # copy config file to out_dir
    shutil.copy(cfg['config_file'], out_dir)

    # create train/test datasets and dataloaders
    data_dir = cfg['data']['path']
    train_set = cystolith_loader(split = 'train',
                                 img_list_file = cfg['data']['train_list'],
                                 img_size = cfg['data']['img_size'],
                                 do_resize = cfg['data']['do_resize'],
                                 img_norm = cfg['data'].get('img_norm',False),
                                 augmentation_params = cfg.get('augmentations',None),
                                 debug_info = cfg['debug_info'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['data']['n_workers'])

    validation_set = cystolith_loader(split = 'validation',
                                      img_list_file = cfg['data']['validation_list'],
                                      img_size = cfg['data']['img_size'],
                                      do_resize = cfg['data']['do_resize'],
                                      img_norm = cfg['data'].get('img_norm',False))
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg['data']['n_workers'])

    # specify device, CUDA if available, CPU otherwise
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # create a neural network
    net = get_net(cfg)
    net.to(device)

    # for param in net.parameters():
    #     print(param)

    # create loss function, optimizer, scheduler according to configuration
    loss_fn = get_loss_function(cfg['train']['loss'])
    optimizer = get_optimizer(cfg['train']['optimizer'], net)
    scheduler = get_scheduler(cfg['train']['scheduler'], optimizer)

    model_to_save = os.path.join(out_dir, cfg['train']['saved_model'])
    best_acc = 0.0
    all_loss = []
    start_epoch = 0
    total_iters = 0

    # load pre-trained model if specified, otherwise start from scratch
    if cfg['train']['pretrained_model'] is not None:
        pretrained_model_path = cfg['train']['pretrained_model']
        if os.path.exists(pretrained_model_path):
            state = torch.load(pretrained_model_path)
            net.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            scheduler.load_state_dict(state['scheduler_state'])
            start_epoch = state['epoch']
            best_acc = state['best_acc']
            all_loss = state['all_loss']
            total_iters = state['total_iters']
            print('Loaded pretrained model')
        else:
            print('Exit - can not load the specified pretrained model')
            exit()
    else:
        print('Start from scratch - no model found')

    # calculate "n_batches_per_validation" - number of iterations for validation
    # note - the reasoning is to perform more than one validation during each epoch
    trainset_size = len(train_set)
    n_batches_per_validation = trainset_size // (cfg['train']['batch_size'] * cfg['train']['n_validations_per_epoch'])

    for epoch in range(start_epoch, start_epoch + cfg['train']['n_epochs']):  # loop over the dataset multiple times
        print("epoch = ", epoch + 1)
        running_loss = 0.0
        # print(len(trainloader))
        for i, data in enumerate(train_loader, 0):
        #for (images, labels) in train_loader:

            inputs, labels = data[0].to(device), data[1].to(device)

            # switch to train mode (note that it has changed during validation)
            net.train()

            # zero the gradients of the network parameters
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i % n_batches_per_validation == n_batches_per_validation - 1):
                # do a validation every 'n_validations_per_epoch'
                total_iters += n_batches_per_validation
                val_idx = (total_iters * cfg['train']['batch_size']) / trainset_size
                curr_acc = validate(net, validation_loader, device)
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / n_batches_per_validation:.3f} loss.item {loss.item():.3f} accuracy {curr_acc:.3f}%')
                curr_loss = [epoch + 1, total_iters, val_idx, running_loss / n_batches_per_validation, curr_acc]
                all_loss.append(curr_loss)
                save_train_loss_info(out_dir, all_loss)

                if curr_acc > best_acc:
                    best_acc = curr_acc
                    print(f'best acc: {best_acc:.2f}')
                    state = {
                        'model_state': net.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'epoch': (epoch+1),
                        'best_acc': best_acc,
                        'all_loss': all_loss,
                        'total_iters': total_iters
                    }
                    torch.save(state, model_to_save)
                running_loss = 0.0

        # adjust total_iters (needed in case trainset_size is not divisible by n_batches_per_validation)
        total_iters = ((epoch + 1) * trainset_size) // cfg['train']['batch_size']

        if scheduler is not None:
            scheduler.step()

    print('Finished train')


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

    # call the train routine
    train(cfg)



