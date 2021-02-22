'''
Metrics to measure calibration of a trained deep neural network.

References:
[1] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks.
    arXiv preprint arXiv:1706.04599, 2017.
'''

import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


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


def expected_calibration_error(confs, preds, labels, num_bins=10):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * \
            abs(bin_accuracy - bin_confidence)
    return ece


def maximum_calibration_error(confs, preds, labels, num_bins=10):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    ce = []
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        ce.append(abs(bin_accuracy - bin_confidence))
    return max(ce)


def average_calibration_error(confs, preds, labels, num_bins=10):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    non_empty_bins = 0
    ace = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        if bin_count > 0:
            non_empty_bins += 1
        ace += abs(bin_accuracy - bin_confidence)
    return ace / float(non_empty_bins)


def l2_error(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    l2_sum = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        l2_sum += (float(bin_count) / num_samples) * \
               (bin_accuracy - bin_confidence)**2
        l2_error = math.sqrt(l2_sum)
    return l2_error


def test_classification_net_logits(logits, labels):
    '''
    This function reports classification accuracy and confusion matrix given logits and labels
    from a model.
    '''
    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    softmax = F.softmax(logits, dim=1)
    confidence_vals, predictions = torch.max(softmax, dim=1)
    labels_list.extend(labels.cpu().numpy().tolist())
    predictions_list.extend(predictions.cpu().numpy().tolist())
    confidence_vals_list.extend(confidence_vals.cpu().numpy().tolist())
    accuracy = accuracy_score(labels_list, predictions_list)
    return confusion_matrix(labels_list, predictions_list), accuracy, labels_list,\
        predictions_list, confidence_vals_list


def test_classification_net(model, data_loader, device):
    '''
    This function reports classification accuracy and confusion matrix over a dataset.
    '''
    model.eval()
    labels_list = []
    predictions_list = []
    confidence_vals_list = []
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            data = data.to(device)
            label = label.to(device)

            logits = model(data)
            softmax = F.softmax(logits, dim=1)
            confidence_vals, predictions = torch.max(softmax, dim=1)

            labels_list.extend(label.cpu().numpy().tolist())
            predictions_list.extend(predictions.cpu().numpy().tolist())
            confidence_vals_list.extend(confidence_vals.cpu().numpy().tolist())
    accuracy = accuracy_score(labels_list, predictions_list)

    return confusion_matrix(labels_list, predictions_list), accuracy, labels_list,\
        predictions_list, confidence_vals_list


# Calibration error scores in the form of loss metrics
class ECELoss(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece


class AdaptiveECELoss(nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15):
        super(AdaptiveECELoss, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class ClasswiseECELoss(nn.Module):
    '''
    Compute Classwise ECE
    '''
    def __init__(self, n_bins=15):
        super(ClasswiseECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1)
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce
    
  
class ClassECELoss(nn.Module):
    '''
    Compute per-Class ECE
    '''
    def __init__(self, n_bins=15):
        super(ClassECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1)
        per_class_sce = None
        choices = torch.argmax(softmaxes, dim=1)
        classes_acc = []

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_choices = choices[labels.eq(i)]
            class_choices = torch.sum(class_choices.eq(i)).item()
            class_accuracy = class_choices / torch.sum(labels.eq(i)).item()
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)
            
            classes_acc.append(class_accuracy)

        return per_class_sce, classes_acc
    
    
# Calibration error scores in the form of loss metrics
class posnegECELoss(nn.Module):
    '''
    Compute per-Class powsitiv and negative ECE
    '''
    def __init__(self, n_bins=15):
        super(posnegECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1)
        per_class_sce = None
        choices = torch.argmax(softmaxes, dim=1)
        classes_acc = []
        
        counts_over = torch.zeros(num_classes)
        counts_under = torch.zeros(num_classes)
        bins_over = []
        bins_under = []

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_choices = choices[labels.eq(i)]
            class_choices = torch.sum(class_choices.eq(i)).item()
            class_accuracy = class_choices / torch.sum(labels.eq(i)).item()
            class_sce_pos = torch.zeros(1, device=logits.device)
            class_sce_neg = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    if avg_confidence_in_bin - accuracy_in_bin > 0:
                        bins_over.append(bin_lower)
                        counts_over[i] += 1 
                        class_sce_pos += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    else:
                        bins_under.append(bin_lower)
                        counts_under[i] += 1
                        class_sce_neg += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce_pos = class_sce_pos
                per_class_sce_neg = class_sce_neg
            else:
                per_class_sce_pos = torch.cat((per_class_sce_pos, class_sce_pos), dim=0)
                per_class_sce_neg = torch.cat((per_class_sce_neg, class_sce_neg), dim=0)
                
            classes_acc.append(class_accuracy)
        print('total samples number: ', labels.shape[0])
        print('over confidence counts sum: ', torch.sum(counts_over).item())
        print('under confidence counts sum: ', torch.sum(counts_under).item())
        print('over confidence bins: ', torch.mean(torch.FloatTensor(bins_over)).item())
        print('under confidence bins: ', torch.mean(torch.FloatTensor(bins_under)).item())

        return per_class_sce_pos, per_class_sce_neg, classes_acc
    
# Calibration error scores in the form of loss metrics
class binsECELoss(nn.Module):
    '''
    Compute per-Class positive and negative ECE
    '''
    def __init__(self, n_bins=15, low_high_bin=0.3):
        super(binsECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.low_high_bin = low_high_bin

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1)
        per_class_sce = None
        choices = torch.argmax(softmaxes, dim=1)
        classes_acc = []
        
        counts_high = torch.zeros(num_classes)
        counts_low = torch.zeros(num_classes)
        high_bins = []
        low_bins = []

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_choices = choices[labels.eq(i)]
            class_choices = torch.sum(class_choices.eq(i)).item()
            class_accuracy = class_choices / torch.sum(labels.eq(i)).item()
            class_sce_high = torch.zeros(1, device=logits.device)
            class_sce_low = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    if bin_lower > self.low_high_bin:
                        high_bins.append(bin_lower)
                        counts_high[i] += 1 
                        class_sce_high += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    else:
                        low_bins.append(bin_lower)
                        counts_low[i] += 1
                        class_sce_low += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce_high = class_sce_high
                per_class_sce_low = class_sce_low
            else:
                per_class_sce_high = torch.cat((per_class_sce_high, class_sce_high), dim=0)
                per_class_sce_low = torch.cat((per_class_sce_low, class_sce_low), dim=0)
                
            classes_acc.append(class_accuracy)
        print('total samples number: ', labels.shape[0])
        print('high bins counts sum: ', torch.sum(counts_high).item())
        print('low bins counts sum: ', torch.sum(counts_low).item())
        print('high bins: ', torch.mean(torch.FloatTensor(high_bins)).item())
        print('low bins: ', torch.mean(torch.FloatTensor(low_bins)).item())

        return per_class_sce_high, per_class_sce_low, classes_acc
    
# Calibration error scores in the form of loss metrics
class diffECELoss(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, n_bins=15):
        super(diffECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        bin_over_confidence = []
        bins = []

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                bin_over_confidence.append(((avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin).item())
                bins.append(bin_lower.item())
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece, bin_over_confidence, bins
    
# Calibration error scores in the form of loss metrics
class estECELoss(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, n_bins=15):
        super(estECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = torch.gather(softmaxes, 1, labels.view(-1,1)).squeeze()
        #accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
    
# Calibration error scores in the form of loss metrics
class posnegECEbinsLoss(nn.Module):
    '''
    Compute per-Class over-confidence and under-confidence ECE vs. bins
    '''
    def __init__(self, n_bins=15):
        super(posnegECEbinsLoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1)
        per_class_sce = None
        
        counts_over = torch.zeros(num_classes)
        counts_under = torch.zeros(num_classes)
        bins_over = []
        bins_under = []
        over_ece_bins = torch.zeros(self.bin_lowers.shape, device=logits.device)
        under_ece_bins = torch.zeros(self.bin_lowers.shape, device=logits.device)
        lower_bin_acc = torch.zeros(num_classes, device=logits.device)
        lower_bin_conf = torch.zeros(num_classes, device=logits.device)
        upper_bin_acc = torch.zeros(num_classes, device=logits.device)
        upper_bin_conf = torch.zeros(num_classes, device=logits.device)
        
        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i
            bin = 0

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    
                    if bin_lower == 0:
                        lower_bin_acc[i] += accuracy_in_bin
                        lower_bin_conf[i] += avg_confidence_in_bin
                    if bin_upper == 1:
                        upper_bin_acc[i] += accuracy_in_bin
                        upper_bin_conf[i] += avg_confidence_in_bin
                        
                    if avg_confidence_in_bin - accuracy_in_bin > 0:
                        over_ece_bins[bin] += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                        bins_over.append(bin_lower)
                        counts_over[i] += 1 
                    else:
                        under_ece_bins[bin] += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                        bins_under.append(bin_lower)
                        counts_under[i] += 1
                bin += 1
                
        print("Lowest bin average accuracy per class: ", lower_bin_acc)
        print("Lowest bin average confidence per class: ", lower_bin_conf)
        print("Upper bin average accuracy per class: ", upper_bin_acc)
        print("Upper bin average confidence per class: ", upper_bin_conf)
                              
        return over_ece_bins, under_ece_bins, self.bin_lowers

