import os
import sys
import torch
import random
import argparse
from torch import nn
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import numpy as np

import matplotlib.pyplot as plt

# Import metrics to compute
from Metrics.metrics import test_classification_net_logits
from Metrics.metrics import ECELoss
from Metrics.metrics2 import ECE, softmax

# Import temperature scaling and NLL utilities
from temperature_scaling import set_temperature2, temperature_scale2, class_temperature_scale2

# Import unpickling logits and labels
from evaluate_scripts.unpickle_probs import unpickle_probs

def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = './'
    model = 'resnet110'
    save_loc = './'
    save_plots_loc = './'
    saved_model_name = 'resnet110_cross_entropy_350.model'
    num_bins = 25
    model_name = None
    train_batch_size = 128
    test_batch_size = 128
    cross_validation_error = 'ece'
    trained_loss = 'cross_entropy'
    logits_path = '/mnt/dsi_vol1/users/frenkel2/data/calibration/trained_models/spline/logits/'
    logits_file = 'probs_resnet110_c10_logits.p'

    parser = argparse.ArgumentParser(
        description="Evaluating a single model on calibration metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to test on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name", help='name of the model')
    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to test')
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to import the model')
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name,
                        dest="saved_model_name", help="file name of the pre-trained model")
    parser.add_argument("--num-bins", type=int, default=num_bins, dest="num_bins",
                        help='Number of bins')
    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("-da", action="store_true", dest="data_aug",
                        help="Using data augmentation")
    parser.set_defaults(data_aug=True)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("--cverror", type=str, default=cross_validation_error,
                        dest="cross_validation_error", help='Error function to do temp scaling')
    parser.add_argument("-log", action="store_true", dest="log",
                        help="whether to print log data")
    parser.add_argument("-acc", action="store_true", dest="acc_check",
                        help="whether to calculate ECE for each class only if accuracy gets better")
    parser.add_argument("-iters", type=int, default=1,
                        dest="temp_opt_iters", help="number of temprature scaling iterations")
    parser.add_argument("-init_temp", type=float, default=2.5,
                        dest="init_temp", help="initial temperature for temperature scaling")
    parser.add_argument("-const_temp", action="store_true", dest="const_temp",
                        help="whether to use constant temperature on all classes")
    parser.add_argument("--save-path-plots", type=str, default=save_plots_loc,
                        dest="save_plots_loc",
                        help='Path to save plots')
    parser.add_argument("--loss", type=str, default=trained_loss,
                        dest="trained_loss",
                        help='Trained loss(cross_entropy/focal_loss/focal_loss_adaptive/mmce/mmce_weighted/brier_score)')
    parser.add_argument("--logits_path", type=str, default=logits_path,
                        dest="logits_path",
                        help='Path of saved logits')
    parser.add_argument("--logits_file", type=str, default=logits_file,
                        dest="logits_file",
                        help='File of saved logits')

    return parser.parse_args()


if __name__ == "__main__":

    # Checking if GPU is available
    cuda = False
    if (torch.cuda.is_available()):
        cuda = True

    # Setting additional parameters
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    args = parseArgs()

    if args.model_name is None:
        args.model_name = args.model

    dataset = args.dataset
    dataset_root = args.dataset_root
    model_name = args.model_name
    save_loc = args.save_loc
    saved_model_name = args.saved_model_name
    num_bins = args.num_bins
    cross_validation_error = args.cross_validation_error
    temp_opt_iters = args.temp_opt_iters
    const_temp = args.const_temp
    save_plots_loc = args.save_plots_loc
    init_temp = args.init_temp
    trained_loss = args.trained_loss
    acc_check = args.acc_check
    logits_file =  args.logits_file
    logits_path = args.logits_path

    ece_criterion = ECELoss(n_bins=25).cuda()
    
    # Loading logits and labels
    file = logits_path + logits_file
    (logits_val, labels_val), (logits_test, labels_test) = unpickle_probs(file)
    
    """
    softmaxs = softmax(logits_test)
    preds = np.argmax(softmaxs, axis=1)
    confs = np.max(softmaxs, axis=1)
    p_ece= ECE(confs, preds, labels_test, bin_size = 1/num_bins) 
    """
    
    logits_val = torch.from_numpy(logits_val).cuda()
    labels_val = torch.squeeze(torch.from_numpy(labels_val), -1).cuda()
    logits_test = torch.from_numpy(logits_test).cuda()
    labels_test = torch.squeeze(torch.from_numpy(labels_test), -1).cuda()

    p_ece = ece_criterion(logits_test, labels_test).item()
    _, p_acc, _, _, _ = test_classification_net_logits(logits_test, labels_test)
    
    # Printing the required evaluation metrics
    if args.log:
        print ('Pre-scaling test ECE: ' + str(p_ece))
        print ('Pre-scaling test accuracy: ' + str(p_acc))

    if const_temp:
        temperature = set_temperature2(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error,
                                       init_temp=init_temp, acc_check=acc_check, const_temp=const_temp, log=args.log, num_bins=25)
    else:                              
        csece_temperature = set_temperature2(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error,
                                                          init_temp=init_temp, acc_check=acc_check, const_temp=const_temp, log=args.log, num_bins=25)
    """
    softmaxs = softmax(class_temperature_scale2(logits_test, csece_temperature))
    preds = np.argmax(softmaxs, axis=1)
    confs = np.max(softmaxs, axis=1)
    ece = ECE(confs, preds, labels_test, bin_size = 1/num_bins)
    """
    ece = ece_criterion(class_temperature_scale2(logits_test, csece_temperature), labels_test).item()
    _, acc, _, _, _ = test_classification_net_logits(class_temperature_scale2(logits_test, csece_temperature), labels_test)
    
    if args.log:
        print ('Post-scaling ECE (Class-based temp scaling): ' + str(ece))
        print ('Post-scaling accuracy: ' + str(acc))
