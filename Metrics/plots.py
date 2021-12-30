'''
This file contains method for generating calibration related plots, eg. reliability plots.

References:
[1] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks.
    arXiv preprint arXiv:1706.04599, 2017.
'''

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import torch
from torch.nn import functional as F
from scipy.interpolate import make_interp_spline
plt.rcParams.update({'font.size': 20})

# Some keys used for the following dictionaries
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'


def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + \
            (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / \
                float(bin_dict[binn][COUNT])
    return bin_dict


def reliability_plot(confs, preds, labels, save_plots_loc, dataset, model, trained_loss, num_bins=15, scaling_related='before', save=False):
    '''
    Method to draw a reliability plot from a model's predictions and confidences.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][BIN_ACC])
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, bns, align='edge', width=0.05, color='pink', label='Expected')
    plt.bar(bns, y, align='edge', width=0.05,
            color='blue', alpha=0.5, label='Actual')
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.legend()
    if save:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'reliability_plot_{}_{}_{}_{}.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    else:
        plt.show()
        
def reliability_plot_chexpert(confs, preds, labels, save_plots_loc, num_bins=15, scaling_related='before', save=False):
    '''
    Method to draw a reliability plot from a model's predictions and confidences.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][BIN_ACC])
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, bns, align='edge', width=0.05, color='pink', label='Expected')
    plt.bar(bns, y, align='edge', width=0.05,
            color='blue', alpha=0.5, label='Actual')
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.legend()
    if save:
        plt.savefig(os.path.join(save_plots_loc, 'chexpert', 'reliability_plot_{}_chexpert.pdf'.format(scaling_related)), dpi=40)
    else:
        plt.show()


def bin_strength_plot(confs, preds, labels, num_bins=15):
    '''
    Method to draw a plot for the number of samples in each confidence bin.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    num_samples = len(labels)
    y = []
    for i in range(num_bins):
        n = (bin_dict[i][COUNT] / float(num_samples)) * 100
        y.append(n)
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, y, align='edge', width=0.05,
            color='blue', alpha=0.5, label='Percentage samples')
    plt.ylabel('Percentage of samples')
    plt.xlabel('Confidence')
    plt.show()
    
def pos_neg_ece_bins_plot(bins_vec, bins_ece_over, bins_ece_under, bins_ece_over_after, bins_ece_under_after, save_plots_loc, dataset, model, trained_loss,
                          acc_check=False, scaling_related='before', const_temp=False):
    plt.figure(figsize=(10, 8))
    plt.scatter(bins_vec, bins_ece_over.cpu(), s=70)
    plt.scatter(bins_vec, bins_ece_under.cpu(), s=70)
    #plt.scatter(bins_vec, bins_ece_over_after.cpu())
    #plt.scatter(bins_vec, bins_ece_under_after.cpu())
    plt.xlabel('bins', fontsize=26)
    plt.xticks(fontsize=18)
    plt.ylabel('ECE', fontsize=26)
    plt.yticks(fontsize=18)
    #plt.legend(('over-confidence classes', 'under-confidence classes', 'over-confidence classes after scaling', 'under-confidence classes after scaling'), fontsize=10)
    plt.legend(('over-confidence classes', 'under-confidence classes'), fontsize=22)
    if const_temp:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'over_under_ece_bins_{}_scaling_{}_{}_{}_const_temp.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    if acc_check:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'over_under_ece_bins_{}_scaling_{}_{}_{}_acc.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    else:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'over_under_ece_bins_{}_scaling_{}_{}_{}.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    plt.close()

def pos_neg_ece_plot(acc, csece_pos, csece_neg, save_plots_loc, dataset, model, trained_loss, acc_check=False, scaling_related='before', const_temp=False):
    plt.figure(figsize=(10, 8))
    plt.scatter(acc, csece_pos.cpu(), s=70)
    plt.xlabel('accuracy', fontsize=26)
    plt.xticks(fontsize=18)
    plt.ylabel('ECE', fontsize=26)
    plt.yticks(fontsize=16)
    plt.ylim(0, 0.01)
    if const_temp:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'pos_ece_acc_{}_scaling_{}_{}_{}_const_temp.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    if acc_check:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'pos_ece_acc_{}_scaling_{}_{}_{}_acc.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    else:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'pos_ece_acc_{}_scaling_{}_{}_{}.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(acc, csece_neg.cpu(), s=70)
    plt.xlabel('accuracy', fontsize=26)
    plt.xticks(fontsize=18)
    plt.ylabel('ECE', fontsize=26)
    plt.yticks(fontsize=16)
    plt.ylim(0, 0.01)
    if const_temp:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'neg_ece_acc_{}_scaling_{}_{}_{}_const_temp.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    if acc_check:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'neg_ece_acc_{}_scaling_{}_{}_{}_acc.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    else:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'neg_ece_acc_{}_scaling_{}_{}_{}.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    plt.close()
    
def ece_acc_plot(acc, csece, save_plots_loc, dataset, model, trained_loss, acc_check=False, scaling_related='before', const_temp=False, unc=False):
    plt.figure(figsize=(10, 8))
    plt.scatter(acc, csece.cpu(), s=70)
    plt.xlabel('accuracy', fontsize=26)
    plt.xticks(fontsize=18)
    plt.ylabel('ECE', fontsize=26)
    plt.yticks(fontsize=16)
    #plt.ylim(0, 0.01)
    if const_temp:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'ece_acc_{}_scaling_{}_{}_{}_const_temp.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    else:
        if acc_check:
            if unc:
                plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'uncalibrated_ece_acc_{}_scaling_{}_{}_{}_acc.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=100)
            else:
                plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'ece_acc_{}_scaling_{}_{}_{}_acc.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
        else:
            if unc:
                plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'uncalibrated_ece_acc_{}_scaling_{}_{}_{}.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
            else:
                plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'ece_acc_{}_scaling_{}_{}_{}.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    plt.close()
    
def ece_iters_plot(scaled_model, save_plots_loc, dataset, model, trained_loss, init_temp, acc_check=False):
    plt.figure()
    plt.plot(range(scaled_model.iters + 1), scaled_model.ece_list)
    plt.plot(range(scaled_model.iters + 1), scaled_model.ece*torch.ones((scaled_model.iters + 1)))
    plt.legend(('class-based temp scaling', 'single temp scaling'), fontsize=10)
    plt.xlabel('iterations', fontsize=10)
    plt.xticks(fontsize=10)
    plt.ylabel('ECE', fontsize=10)
    plt.yticks(fontsize=10)
    if acc_check:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'ece_iters_{}_{}_{}_{}_acc.pdf'.format(init_temp, dataset, model, trained_loss)), dpi=40)
    else:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'ece_iters_{}_{}_{}_{}.pdf'.format(init_temp, dataset, model, trained_loss)), dpi=40)
    plt.close()
    
def temp_acc_plot(acc, temp, single_temp, save_plots_loc, dataset, model, trained_loss, acc_check=False, const_temp=False):
    plt.figure()
    plt.scatter(acc, temp.cpu(), label='Class-based temperature')
    plt.plot(acc, single_temp*torch.ones(len(acc)), color='red', label='Single temperature')
    plt.xlabel('accuracy', fontsize=10)
    plt.xticks(fontsize=10)
    plt.ylabel('Temperature', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    if const_temp:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'temp_acc_after_scaling_{}_{}_{}_const_temp.pdf'.format(dataset, model, trained_loss)), dpi=40)
    else:
        if acc_check:
            plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'temp_acc_after_scaling_{}_{}_{}_acc.pdf'.format(dataset, model, trained_loss)), dpi=40)
        else:
            plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'temp_acc_after_scaling_{}_{}_{}.pdf'.format(dataset, model, trained_loss)), dpi=40)
            
def temp_acc_plot_chexpert(acc, temp, single_temp, save_plots_loc, acc_check=False, const_temp=False):
    plt.figure()
    plt.scatter(acc, temp.cpu(), label='Class-based temperature')
    plt.plot(acc, single_temp*torch.ones(len(acc)), color='red', label='Single temperature')
    plt.xlabel('accuracy', fontsize=10)
    plt.xticks(fontsize=10)
    plt.ylabel('Temperature', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    if const_temp:
        plt.savefig(os.path.join(save_plots_loc, 'chexpert', 'temp_acc_after_scaling_{}_{}_{}_const_temp_chexpert.pdf'), dpi=40)
    else:
        if acc_check:
            plt.savefig(os.path.join(save_plots_loc, 'chexpert', 'temp_acc_after_scaling_{}_{}_{}_acc_chexpert.pdf'), dpi=40)
        else:
            plt.savefig(os.path.join(save_plots_loc, 'chexpert', 'temp_acc_after_scaling_{}_{}_{}_chexpert.pdf'), dpi=40)


def diff_ece_plot(acc, csece1, csece2, save_plots_loc, dataset, model, trained_loss, acc_check=False, scaling_type='class_based'):
    plt.figure()
    plt.scatter(acc, (csece1 - csece2).cpu())
    plt.xlabel('accuracy', fontsize=10)
    plt.xticks(fontsize=10)
    plt.ylabel('ECE difference', fontsize=10)
    plt.yticks(fontsize=10)
    plt.axhline(y=0, color='r')
    if acc_check:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'diff_{}_ece_acc_after_scaling_{}_{}_{}_acc.pdf'.format(scaling_type, dataset, model, trained_loss)), dpi=40)
    else:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'diff_{}_ece_acc_after_scaling_{}_{}_{}.pdf'.format(scaling_type, dataset, model, trained_loss)), dpi=40)


def bins_over_conf_plot(bins, diff, save_plots_loc, dataset, model, trained_loss, scaling_related='before'):
    plt.figure()
    plt.plot(bins, diff)
    plt.xlabel('bins', fontsize=10)
    plt.xticks(fontsize=10)
    plt.ylabel('confidence - accuracy', fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'over_conf_bins_{}_scaling_{}_{}_{}.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)


def temp_bins_plot(single_T, bins_T, bin_boundaries, save_plots_loc, dataset, model, trained_loss, acc_check=False, const_temp=False, divide='reg_divide', ds='val', version=1, cross_validate='ECE', y_name='Temperature'):
    bin_boundaries = torch.linspace(0, bins_T.shape[0], bins_T.shape[0] + 1)
    bin_lowers = bin_boundaries[:-1]
    plt.figure()
    for i in range(bins_T.shape[1]):
        #bin_lowers = bin_boundaries[i][:-1]
        #x_new = np.linspace(1, bins_T.shape[0], 300)
        #a_BSpline = make_interp_spline(bin_lowers, bins_T[:, i].cpu())
        #y_new = a_BSpline(x_new)
        plt.plot(bin_lowers, bins_T[:, i].cpu(), label='Iteration #{}'.format(i + 1))
        #plt.plot(x_new, y_new, label='CBT ({})'.format(cross_validate))
        #plt.plot(x_new, y_new, label='Iteration #{}'.format(i + 1))
    #plt.plot(bin_lowers, torch.ones(bins_T.shape[0])*single_T, label='Single temperature')
    #plt.plot(x_new, torch.ones(len(y_new)) * single_T, label='TS'.format(cross_validate))
    plt.xlabel('Bins', fontsize=16)
    plt.xticks(fontsize=10)
    plt.ylabel(y_name, fontsize=16)
    plt.yticks(fontsize=10)
    # plt.legend(fontsize=14)
    plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'temp_bins_{}_iters_{}_{}_{}_ver_{}_{}_{}_{}_smooth.pdf'.format(bins_T.shape[1], dataset, model, trained_loss, version, divide, ds, cross_validate)), dpi=40)
    
def temp_bins_plot_chexpert(single_T, bins_T, bin_boundaries, save_plots_loc, acc_check=False, const_temp=False, divide='reg_divide', ds='val', version=1, cross_validate='ECE', y_name='Temperature'):
    bin_boundaries = torch.linspace(0, bins_T.shape[0], bins_T.shape[0] + 1)
    bin_lowers = bin_boundaries[:-1]
    plt.figure()
    for i in range(bins_T.shape[1]):
        #bin_lowers = bin_boundaries[i][:-1]
        #x_new = np.linspace(1, bins_T.shape[0], 300)
        #a_BSpline = make_interp_spline(bin_lowers, bins_T[:, i].cpu())
        #y_new = a_BSpline(x_new)
        plt.plot(bin_lowers, bins_T[:, i].cpu(), label='Iteration #{}'.format(i + 1))
        #plt.plot(x_new, y_new, label='CBT ({})'.format(cross_validate))
        #plt.plot(x_new, y_new, label='Iteration #{}'.format(i + 1))
    #plt.plot(bin_lowers, torch.ones(bins_T.shape[0])*single_T, label='Single temperature')
    #plt.plot(x_new, torch.ones(len(y_new)) * single_T, label='TS'.format(cross_validate))
    plt.xlabel('Bins', fontsize=16)
    plt.xticks(fontsize=10)
    plt.ylabel(y_name, fontsize=16)
    plt.yticks(fontsize=10)
    # plt.legend(fontsize=14)
    plt.savefig(os.path.join(save_plots_loc, 'chexpert', 'temp_bins_{}_iters_ver_{}_{}_{}_{}_smooth_chexpert.pdf'.format(bins_T.shape[1], version, divide, ds, cross_validate)), dpi=40)


def ece_bin_plot(ece_bin, single_ece_bin, origin_ece_bin, save_plots_loc, dataset, model, trained_loss, divide='reg_divide', ds='val', version=1):
    plt.figure()
    origin_ece_bin = [i * 100 for i in origin_ece_bin]
    single_ece_bin = [i * 100 for i in single_ece_bin]
    ece_bin = [i * 100 for i in ece_bin]
    plt.plot(range(len(ece_bin)), origin_ece_bin, label='ECE before scaling')
    plt.plot(range(len(ece_bin)), single_ece_bin, label='ECE after single temp scaling')
    plt.plot(range(len(ece_bin)), ece_bin, label='ECE after per bin temp scaling')
    plt.xlabel('Bins', fontsize=16)
    plt.xticks(fontsize=10)
    plt.ylabel('ECE(%)', fontsize=16)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model),
                             'ece_bins_{}_{}_{}_ver_{}_{}_{}_smooth.pdf'.format(dataset, model, trained_loss, version,
                                                                                divide, ds)), dpi=40)
    
def ece_bin_plot_chexpert(ece_bin, single_ece_bin, origin_ece_bin, save_plots_loc, divide='reg_divide', ds='val', version=1):
    plt.figure()
    origin_ece_bin = [i * 100 for i in origin_ece_bin]
    single_ece_bin = [i * 100 for i in single_ece_bin]
    ece_bin = [i * 100 for i in ece_bin]
    plt.plot(range(len(ece_bin)), origin_ece_bin, label='ECE before scaling')
    plt.plot(range(len(ece_bin)), single_ece_bin, label='ECE after single temp scaling')
    plt.plot(range(len(ece_bin)), ece_bin, label='ECE after per bin temp scaling')
    plt.xlabel('Bins', fontsize=16)
    plt.xticks(fontsize=10)
    plt.ylabel('ECE(%)', fontsize=16)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(save_plots_loc, 'chexpert',
                             'ece_bins_ver_{}_{}_{}_smooth_chexpert.pdf'.format(version,
                                                                                divide, ds)), dpi=40)
    
    
def logits_diff_bin_plot(logits_diff_bin, save_plots_loc, dataset, model, trained_loss, divide='reg_divide', ds='val', version=1):
    plt.figure()
    plt.plot(range(len(logits_diff_bin)), logits_diff_bin)
    plt.xlabel('Bins', fontsize=10)
    plt.xticks(fontsize=10)
    plt.ylabel('Logits difference', fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model),
                            'logits_diff_bins_{}_{}_{}_ver_{}_{}_{}.pdf'.format(dataset, model, trained_loss, version,
                                                                                divide, ds)), dpi=40)
    
    
def temp_bins_plot2(single_T, single_T2, bins_T, bins_T2, bin_boundaries, bin_boundaries2, save_plots_loc, dataset, model, trained_loss, divide='reg_divide', ds='val', version=1, y_name='Temperature'):
    bin_boundaries = torch.linspace(0, bins_T.shape[0], bins_T.shape[0] + 1)
    bin_lowers = bin_boundaries[:-1]
    plt.figure()
    for i in range(bins_T.shape[1]):
        #bin_lowers = bin_boundaries[i][:-1]
        #bin_lowers2 = bin_boundaries2[i][:-1]
        # x_new = np.linspace(1, bins_T.shape[0], 300)
        # a_BSpline = make_interp_spline(bin_lowers, bins_T[:, i].cpu())
        # a_BSpline2 = make_interp_spline(bin_lowers, bins_T2[:, i].cpu())
        # y_new = a_BSpline(x_new)
        # y_new2 = a_BSpline2(x_new)
        plt.plot(bin_lowers, bins_T[:, i].cpu(), label='Weights')
        plt.plot(bin_lowers, (1 / bins_T2[:, i]).cpu(), label=r'$1/Temperatures$')
        # plt.plot(x_new, y_new, label='CBT ResNet-152')
        # plt.plot(x_new, y_new2, label='CBT DenseNet-161')
        #plt.plot(x_new, y_new, label='Iteration #{}'.format(i))
    #plt.plot(bin_lowers, torch.ones(bins_T.shape[0])*single_T, label='Single temperature')
    # plt.plot(x_new, torch.ones(len(y_new)) * single_T, label='TS ResNet-152')
    # plt.plot(x_new, torch.ones(len(y_new2)) * single_T2, label='TS DenseNet-161')
    plt.xlabel('Bins', fontsize=16)
    plt.xticks(fontsize=10)
    plt.ylabel(y_name, fontsize=16)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'temp_bins_{}_iters_{}_{}_{}_ver_{}_{}_{}_smooth.pdf'.format(bins_T.shape[1], dataset, model, trained_loss, version, divide, ds)), dpi=40)
    

def exp_value(confidences, diff):
    numerator = (-1 + torch.sqrt(1 + 4 * (1 - confidences) / confidences)) / 2  
    denominator = (-1 + torch.sqrt(1 + 4 * (1 - (confidences - diff)) / (confidences - diff))) / 2 
    
    return numerator, denominator


def plot_temp_different_bins(save_plots_loc):
    confidences = torch.linspace(0.61, 1, 40)
    #optim_temps = torch.log((1 - confidences) / confidences) / torch.log((1 - (confidences - 0.1)) / (confidences - 0.1))
    numerator1, denominator1 = exp_value(confidences, 0.1)
    numerator2, denominator2 = exp_value(confidences, 0.05)
    numerator3, denominator3 = exp_value(confidences, 0.03)
    #numerator4, denominator4 = exp_value(confidences, 0.2)
    optim_temps1 = torch.log(numerator1) / torch.log(denominator1)
    optim_temps2 = torch.log(numerator2) / torch.log(denominator2)
    optim_temps3 = torch.log(numerator3) / torch.log(denominator3)
    #optim_temps4 = torch.log(numerator4) / torch.log(denominator4)
    plt.figure()
    #plt.plot(confidences, optim_temps4, label='\u03B5=0.2')
    plt.plot(confidences, optim_temps1, label='\u03B5=0.1')
    plt.plot(confidences, optim_temps2, label='\u03B5=0.05')
    plt.plot(confidences, optim_temps3, label='\u03B5=0.03')
    plt.xlabel('Confidence', fontsize=16)
    plt.xticks(fontsize=10)
    plt.ylabel('Temperature', fontsize=16)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=14)
    plt.savefig(os.path.join(save_plots_loc, 'temp_movements_between_bins_3_classes.pdf'), dpi=40)
    

def ece_iters_plot2(single_ece, single_ece2, ece_list1, ece_list2, save_plots_loc, dataset, model, trained_loss, divide='reg_divide', ds='val', version=1):
    if len(ece_list1) < len(ece_list2):
        ece_list1 = ece_list1 + (len(ece_list2) - len(ece_list1)) * [ece_list1[-1]]
    elif len(ece_list1) > len(ece_list2):
        ece_list2 = ece_list2 + (len(ece_list1) - len(ece_list2)) * [ece_list2[-1]]
    ece_list1 = [i * 100 for i in ece_list1]
    ece_list2 = [i * 100 for i in ece_list2]
    plt.figure()
    plt.plot(range(len(ece_list1)), ece_list1, label='CBT ResNet-152')
    plt.plot(range(len(ece_list2)), ece_list2, label='CBT DenseNet-161')
    plt.plot(range(len(ece_list1)), torch.ones(len(ece_list1)) * single_ece, label='TS ResNet-152')
    plt.plot(range(len(ece_list2)), torch.ones(len(ece_list2)) * single_ece2, label='TS DenseNet-161')
    plt.xlabel('Iterations', fontsize=16)
    plt.xticks(fontsize=10)
    plt.ylabel('ECE(%)', fontsize=16)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=14)
    plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'ece_iters_{}_iters_{}_{}_{}_ver_{}_{}_{}_smooth.pdf'.format(len(ece_list1) - 1, dataset, model, trained_loss, version, divide, ds)), dpi=40)
    
def plot_trajectory(save_plots_loc):  # For probabilities [0.6, 0.3, 0.1]
    weights = torch.linspace(0, 1, 100).unsqueeze(-1)
    temperatures = torch.linspace(1, 100, 10000).unsqueeze(-1)
    starting_point = torch.tensor([0.6, 0.3]).unsqueeze(0)
    starting_logits = torch.tensor([math.log(0.6), math.log(0.3), math.log(0.1)])
    # starting_logits = torch.tensor([2.2, 1.525, 0.5])
    ts_points = [F.softmax(starting_logits / temperature, dim=0) for temperature in temperatures]
    ts_points = torch.stack(ts_points)
    n_classes = starting_point.shape[1] + 1
    ws_points = torch.matmul(weights, (1 / n_classes) * torch.ones(starting_point.shape)) + torch.matmul(1 - weights, starting_point)
    ws_points_full = torch.cat((ws_points, (1 - torch.sum(ws_points, 1)).unsqueeze(-1)), 1)
    weights_ent = -torch.sum(ws_points_full * torch.log2(ws_points_full), 1)
    softmaxes_100 = torch.tensor([8.4042679500e-13, 1.4278050742e-08, 3.9925965312e-11, 7.8529644267e-14,
        1.1687384394e-10, 9.7083494401e-14, 7.9007286824e-13, 1.1496912363e-13,
        5.3773496073e-12, 7.6878958755e-10, 8.9035365747e-09, 5.3947623278e-12,
        2.4426896617e-10, 2.2383541201e-11, 1.2707822294e-10, 2.1816673468e-10,
        5.0172353387e-15, 1.6286461112e-12, 5.1560413925e-12, 8.6647043707e-12,
        1.8531972623e-09, 2.7630087107e-10, 7.1155463308e-16, 3.7386840152e-11,
        5.1252758981e-11, 3.1181262433e-11, 2.6755674298e-06, 9.9959415197e-01,
        1.9884007635e-11, 1.1077156523e-04, 1.7637266647e-11, 2.2995503279e-09,
        7.3481587606e-06, 1.2129663940e-09, 3.2103027479e-05, 5.2368401282e-11,
        2.3453745612e-09, 2.9135565488e-11, 2.9145277771e-12, 3.5043259961e-11,
        9.6558103581e-14, 1.9227650583e-09, 1.5236486206e-07, 4.5127812598e-09,
        8.7795990112e-05, 3.4632095776e-05, 3.3900747098e-08, 5.3773188159e-12,
        4.9334299666e-13, 4.7792599739e-11, 9.7179556069e-12, 2.9196653486e-05,
        1.2558685400e-15, 1.9376671101e-10, 2.1402189916e-12, 1.7672345792e-12,
        4.2892519397e-11, 8.4134947273e-12, 1.5762311595e-11, 2.2964830992e-12,
        1.1481499413e-14, 4.4955605211e-11, 2.6382507290e-11, 1.0882557433e-07,
        3.2325153665e-10, 1.4755903444e-10, 2.8219235976e-11, 1.1946493714e-06,
        5.6229808136e-12, 4.9992823214e-09, 1.2134488726e-11, 2.2948927203e-09,
        1.0463446776e-09, 2.0963939562e-07, 1.3484322992e-08, 1.1520114862e-09,
        1.9648471489e-13, 6.5380464775e-07, 2.2771805561e-06, 6.8640011210e-12,
        2.4578919692e-05, 2.0577129952e-13, 2.1242145684e-13, 2.3415527872e-13,
        4.5339165755e-10, 4.0936140522e-07, 9.8099343132e-16, 9.6455538001e-11,
        4.4561368484e-11, 4.3079886880e-10, 1.0865559563e-09, 7.0311572927e-05,
        6.6880915140e-14, 4.8056293167e-08, 3.0499626199e-16, 5.0754581093e-11,
        4.9211958293e-12, 9.5986638371e-07, 1.9191167766e-08, 1.8387422074e-07]).unsqueeze(0)
    ws_points2 = torch.matmul(weights, (1 / n_classes) * torch.ones(softmaxes_100.shape)) + torch.matmul(1 - weights, softmaxes_100)
    weights_ent2 = -torch.sum(ws_points2 * torch.log2(ws_points2), 1)
    plt.figure()
    plt.plot(ws_points[:, 0], ws_points[:, 1], label='Weight Scaling')
    plt.plot(ts_points[:, 0], ts_points[:, 1], label='Temperature Scaling')
    plt.xlabel(r'$p_1$', fontsize=16)
    plt.xticks(fontsize=10)
    plt.ylabel(r'$p_2$', fontsize=16)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(save_plots_loc, 'trajectories.pdf'), dpi=40)
    plt.close()
    
    plt.figure()
    plt.plot(ws_points[:, 0], weights_ent)
    plt.xlabel(r'$p_1$', fontsize=16)
    plt.xticks(fontsize=10)
    plt.ylabel('Entropy', fontsize=16)
    plt.yticks(fontsize=10)
    plt.savefig(os.path.join(save_plots_loc, 'entropy.pdf'), dpi=40)
    
    plt.figure()
    plt.plot(ws_points2.max(1)[0], weights_ent2)
    plt.xlabel('Confidence', fontsize=16)
    plt.xticks(fontsize=10)
    plt.ylabel('Entropy', fontsize=16)
    plt.yticks(fontsize=10)
    plt.savefig(os.path.join(save_plots_loc, 'entropy_100.pdf'), dpi=40)

def conf_acc_diff_plot(conf_acc_diff, save_plots_loc, dataset, model, trained_loss, divide='reg_divide', ds='val', version=1):
    plt.figure()
    plt.plot(range(len(conf_acc_diff)), conf_acc_diff)
    plt.xlabel('Bins', fontsize=16)
    plt.xticks(fontsize=10)
    plt.ylabel('Confidence - Accuracy', fontsize=16)
    plt.yticks(fontsize=10)
    plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'conf_acc_diff_bins_{}_{}_{}_{}_ver_{}_{}_{}.pdf'.format(len(conf_acc_diff), dataset, model, trained_loss, version, divide, ds)), dpi=40)
    