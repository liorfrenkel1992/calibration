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
from Metrics.plots import temp_bins_plot, ece_bin_plot, logits_diff_bin_plot, reliability_plot, temp_bins_plot2
from Metrics.plots import plot_temp_different_bins, ece_iters_plot2

# Import temperature scaling and NLL utilities
from temperature_scaling import set_temperature2, temperature_scale2, class_temperature_scale2, set_temperature3, bins_temperature_scale_test3, set_temperature4
from temperature_scaling import bins_temperature_scale_test4, bins_temperature_scale_test5, set_temperature5

# Import unpickling logits and labels
from evaluate_scripts.unpickle_probs import unpickle_probs

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = './'
    model = 'resnet110'
    save_loc = './'
    save_plots_loc = './'
    saved_model_name = 'resnet110_cross_entropy_350.model'
    num_bins = 35
    model_name = None
    train_batch_size = 128
    test_batch_size = 128
    cross_validation_error = 'ece'
    trained_loss = 'cross_entropy'
    logits_path = '/mnt/dsi_vol1/users/frenkel2/data/calibration/trained_models/spline/logits/'
    #logits_path = 'C:/Users/liorf/OneDrive - Bar-Ilan University/calibration/trained_models/spline/logits/'
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
    parser.add_argument("-bins", action="store_true", dest="bins_temp",
                        help="whether to calculate ECE for each bin separately")
    parser.add_argument("-dists", action="store_true", dest="dists",
                        help="whether to optimize ECE by dists from uniform probability")
    parser.add_argument("--divide", type=str, default="equal_divide", dest="divide",
                        help="How to divide bins (reg/equal)")

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
    logits_file = args.logits_file
    #logits_file1 = 'probs_resnet152_imgnet_logits.p'
    #logits_file2 = 'probs_densenet161_imgnet_logits.p'
    logits_path = args.logits_path


    ece_criterion = ECELoss(n_bins=25).cuda()
    
    # Loading logits and labels
    file = logits_path + logits_file
    #file1 = logits_path + logits_file1
    #file2 = logits_path + logits_file2
    (logits_val, labels_val), (logits_test, labels_test) = unpickle_probs(file)
    #(logits_val1, labels_val1), (logits_test1, labels_test1) = unpickle_probs(file1)
    #(logits_val2, labels_val2), (logits_test2, labels_test2) = unpickle_probs(file2)
    
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

    """
    logits_val1 = torch.from_numpy(logits_val1).cuda()
    labels_val1 = torch.squeeze(torch.from_numpy(labels_val1), -1).cuda()
    logits_test1 = torch.from_numpy(logits_test1).cuda()
    labels_test1 = torch.squeeze(torch.from_numpy(labels_test1), -1).cuda()
    logits_val2= torch.from_numpy(logits_val2).cuda()
    labels_val2 = torch.squeeze(torch.from_numpy(labels_val2), -1).cuda()
    logits_test2 = torch.from_numpy(logits_test2).cuda()
    labels_test2 = torch.squeeze(torch.from_numpy(labels_test2), -1).cuda()
    """
    #before_indices, after_indices = check_movements(logits_val, const=2)
    #plot_temp_different_bins(save_plots_loc)
    
    p_ece = ece_criterion(logits_test, labels_test).item()
    #p_ece2 = ece_criterion(logits_test2, labels_test2).item()
    _, p_acc, _, predictions, confidences = test_classification_net_logits(logits_test, labels_test)
    reliability_plot(confidences, predictions, labels_test, save_plots_loc, dataset, args.model, trained_loss, num_bins=num_bins, scaling_related='before', save=True)
    
    # Printing the required evaluation metrics
    if args.log:
        print('Pre-scaling test ECE: ' + str(p_ece))
        print('Pre-scaling test accuracy: ' + str(p_acc))

    if args.dists:
        bins_T, single_temp, bin_boundaries, best_iter = set_temperature4(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error, init_temp=init_temp,
                                                                                         acc_check=acc_check, const_temp=const_temp, log=args.log, num_bins=num_bins, top_temp=1.2)
        # temperature = set_temperature5(logits_val, labels_val, log=args.log)
    
    elif args.bins_temp:
        _, _, _, predictions, confidences = test_classification_net_logits(logits_test, labels_test)
        reliability_plot(confidences, predictions, labels_test, save_plots_loc, dataset, args.model, trained_loss, num_bins=num_bins, scaling_related='before', save=True)
        if const_temp:
            temperature = set_temperature3(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error,
                                        init_temp=init_temp, const_temp=const_temp, log=args.log, num_bins=num_bins)
        else:                              
            bins_T, single_temp, bin_boundaries, many_samples, best_iter = set_temperature3(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error, init_temp=init_temp,
                                                                                                    acc_check=acc_check, const_temp=const_temp, log=args.log, num_bins=num_bins, top_temp=1.2)
            #bins_T2, single_temp2, bin_boundaries2, many_samples2, best_iter2 = set_temperature3(logits_val2, labels_val2, temp_opt_iters, cross_validate=cross_validation_error, init_temp=init_temp,
            #                                                                                    acc_check=acc_check, const_temp=const_temp, log=args.log, num_bins=num_bins, top_temp=1.2)
        
    else:    
        if const_temp:
            temperature = set_temperature2(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error,
                                        init_temp=init_temp, acc_check=acc_check, const_temp=const_temp, log=args.log, num_bins=num_bins)
        else:                              
            csece_temperature, single_temp = set_temperature2(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error,
                                                            init_temp=init_temp, acc_check=acc_check, const_temp=const_temp, log=args.log, num_bins=num_bins)
    
    """
    softmaxs = softmax(class_temperature_scale2(logits_test, csece_temperature))
    preds = np.argmax(softmaxs, axis=1)
    confs = np.max(softmaxs, axis=1)
    ece = ECE(confs, preds, labels_test, bin_size = 1/num_bins)
    """
    
    if args.dists:
        new_softmaxes, ece_bin, single_ece_bin, origin_ece_bin, ece_list = bins_temperature_scale_test4(logits_test, labels_test, bins_T,
                                                                                                        args.temp_opt_iters,
                                                                                                        bin_boundaries,
                                                                                                        single_temp, best_iter, num_bins)
        bins_T2, single_temp2, bin_boundaries2, many_samples, best_iter2 = set_temperature3(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error, init_temp=init_temp,
                                                                                        const_temp=const_temp, log=args.log, num_bins=num_bins)
        # temp_bins_plot(single_temp, bins_T, bin_boundaries, save_plots_loc, dataset, args.model, trained_loss,
        #                divide=args.divide, ds='val_dists', version=2, cross_validate=cross_validation_error, y_name='Weight')
        temp_bins_plot2(single_temp, single_temp2, bins_T, bins_T2, bin_boundaries, bin_boundaries2, save_plots_loc, dataset,
                        args.model, trained_loss, divide=args.divide, ds='val_dists_bins', version=2, y_name='Log Temperature / Weight')
        _, _, _, predictions, confidences = test_classification_net_logits(new_softmaxes, labels_test, is_logits=True)
        reliability_plot(confidences, predictions, labels_test, save_plots_loc, dataset, args.model, trained_loss, num_bins=num_bins, scaling_related='after', save=True)
        # new_softmaxes = bins_temperature_scale_test5(logits_test, temperature)
        _, _, _, predictions, confidences = test_classification_net_logits(temperature_scale2(logits_test, single_temp), labels_test)
        reliability_plot(confidences, predictions, labels_test, save_plots_loc, dataset, args.model, trained_loss, num_bins=num_bins, scaling_related='after_const', save=True)
        ece = ece_criterion(new_softmaxes, labels_test, is_logits=False).item()
    
    elif args.bins_temp:
        temp_bins_plot(single_temp, bins_T, bin_boundaries, save_plots_loc, dataset, args.model, trained_loss,
                       divide=args.divide, ds='val', version=2, cross_validate=cross_validation_error)
        #temp_bins_plot2(single_temp, single_temp2, bins_T, bins_T2, bin_boundaries, bin_boundaries2, save_plots_loc, dataset, args.model, 
        #                trained_loss, divide=args.divide, ds='val_two_models', version=2)
        scaled_logits, ece_bin, single_ece_bin, origin_ece_bin, ece_list = bins_temperature_scale_test3(logits_test, labels_test, bins_T,
                                                                                                        args.temp_opt_iters,
                                                                                                        bin_boundaries, many_samples,
                                                                                                        single_temp, best_iter, num_bins)
        #scaled_logits2, ece_bin2, single_ece_bin2, origin_ece_bin2, ece_list2 = bins_temperature_scale_test3(logits_test2, labels_test2, bins_T2,
        #                                                                                                    args.temp_opt_iters,
        #                                                                                                    bin_boundaries2, many_samples2,
        #                                                                                                    single_temp2, best_iter2, num_bins)
        #ece_list = [p_ece] + ece_list
        #ece_list2 = [p_ece2] + ece_list2
        ece_single = ece_criterion(temperature_scale2(logits_test, single_temp), labels_test).item()
        #ece_single2 = ece_criterion(temperature_scale2(logits_test, single_temp2), labels_test).item()
        #ece_iters_plot2(ece_single, ece_single2, ece_list, ece_list2, save_plots_loc, dataset, args.model, 
        #                trained_loss, divide=args.divide, ds='val_two_models_iters', version=2)
        _, _, _, predictions, confidences = test_classification_net_logits(scaled_logits, labels_test)
        reliability_plot(confidences, predictions, labels_test, save_plots_loc, dataset, args.model, trained_loss, num_bins=num_bins, scaling_related='after', save=True)
        ece_bin_plot(ece_bin, single_ece_bin, origin_ece_bin, save_plots_loc, dataset, args.model, trained_loss,
                       divide=args.divide, ds='test', version=2)
        ece = ece_criterion(scaled_logits, labels_test).item()
    else:
        ece = ece_criterion(class_temperature_scale2(logits_test, csece_temperature), labels_test).item()
    # _, _, _, predictions, confidences = test_classification_net_logits(temperature_scale2(logits_test, single_temp), labels_test)
    # reliability_plot(confidences, predictions, labels_test, save_plots_loc, dataset, args.model, trained_loss, num_bins=num_bins, scaling_related='after_single', save=True)
    #_, acc, _, _, _ = test_classification_net_logits(class_temperature_scale2(logits_test, csece_temperature), labels_test)
    
    if args.log:
        print ('Post-scaling ECE (Class-based temp scaling): ' + str(ece))
        #print ('Post-scaling ECE (Single temp scaling): ' + str(ece_single))
        #print ('Post-scaling accuracy: ' + str(acc))
        
