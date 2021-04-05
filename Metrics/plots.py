'''
This file contains method for generating calibration related plots, eg. reliability plots.

References:
[1] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks.
    arXiv preprint arXiv:1706.04599, 2017.
'''

import math
import matplotlib.pyplot as plt
import os
import torch
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
    plt.figure()
    plt.scatter(bins_vec, bins_ece_over.cpu())
    plt.scatter(bins_vec, bins_ece_under.cpu())
    #plt.scatter(bins_vec, bins_ece_over_after.cpu())
    #plt.scatter(bins_vec, bins_ece_under_after.cpu())
    plt.xlabel('bins', fontsize=12)
    plt.xticks(fontsize=10)
    plt.ylabel('ECE', fontsize=12)
    plt.yticks(fontsize=10)
    #plt.legend(('over-confidence classes', 'under-confidence classes', 'over-confidence classes after scaling', 'under-confidence classes after scaling'), fontsize=10)
    plt.legend(('over-confidence classes', 'under-confidence classes'), fontsize=10)
    if const_temp:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'over_under_ece_bins_{}_scaling_{}_{}_{}_const_temp.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    if acc_check:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'over_under_ece_bins_{}_scaling_{}_{}_{}_acc.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    else:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'over_under_ece_bins_{}_scaling_{}_{}_{}.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    plt.close()

def pos_neg_ece_plot(acc, csece_pos, csece_neg, save_plots_loc, dataset, model, trained_loss, acc_check=False, scaling_related='before', const_temp=False):
    plt.figure()
    plt.scatter(acc, csece_pos.cpu())
    plt.xlabel('accuracy', fontsize=12)
    plt.xticks(fontsize=10)
    plt.ylabel('ECE', fontsize=12)
    plt.yticks(fontsize=10)
    #plt.ylim(0, 0.01)
    if const_temp:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'pos_ece_acc_{}_scaling_{}_{}_{}_const_temp.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    if acc_check:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'pos_ece_acc_{}_scaling_{}_{}_{}_acc.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    else:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'pos_ece_acc_{}_scaling_{}_{}_{}.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    plt.close()

    plt.figure()
    plt.scatter(acc, csece_neg.cpu())
    plt.xlabel('accuracy', fontsize=12)
    plt.xticks(fontsize=10)
    plt.ylabel('ECE', fontsize=12)
    plt.yticks(fontsize=10)
    #plt.ylim(0, 0.01)
    if const_temp:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'neg_ece_acc_{}_scaling_{}_{}_{}_const_temp.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    if acc_check:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'neg_ece_acc_{}_scaling_{}_{}_{}_acc.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    else:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'neg_ece_acc_{}_scaling_{}_{}_{}.pdf'.format(scaling_related, dataset, model, trained_loss)), dpi=40)
    plt.close()
    
def ece_acc_plot(acc, csece, save_plots_loc, dataset, model, trained_loss, acc_check=False, scaling_related='before', const_temp=False, unc=False):
    plt.figure()
    plt.scatter(acc, csece.cpu())
    plt.xlabel('accuracy', fontsize=12)
    plt.xticks(fontsize=10)
    plt.ylabel('ECE', fontsize=12)
    plt.yticks(fontsize=10)
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

def temp_bins_plot(single_T, bins_T, bin_boundaries, save_plots_loc, dataset, model, trained_loss, acc_check=False, const_temp=False, version=1):
    bin_boundaries = torch.linspace(0, bins_T.shape[0], bins_T.shape[0] + 1)
    bin_lowers = bin_boundaries[:-1]
    plt.figure()
    for i in range(bins_T.shape[1]):
        plt.plot(bin_lowers, bins_T[:, i].cpu(), label='iter number {}'.format(i+1))
    plt.plot(bin_lowers, torch.ones(bins_T.shape[0])*single_T, label='Single temperature')
    plt.xlabel('Bins', fontsize=10)
    plt.xticks(fontsize=10)
    plt.ylabel('Temperature', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    if const_temp:
        plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'temp_bins_{}_iters_{}_{}_{}_const_temp_ver_{}.pdf'.format(bins_T.shape[1], dataset, model, trained_loss, version)), dpi=40)
    else:
        if acc_check:
            plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'temp_bins_{}_iters_{}_{}_{}_acc_ver_{}.pdf'.format(bins_T.shape[1], dataset, model, trained_loss, version)), dpi=40)
        else:
            plt.savefig(os.path.join(save_plots_loc, '{}_{}'.format(dataset, model), 'temp_bins_{}_iters_{}_{}_{}_ver_{}.pdf'.format(bins_T.shape[1], dataset, model, trained_loss, version)), dpi=40)