import torch
from torch import nn
from torch.nn import functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


class ECELoss(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.n_bins = n_bins

    def forward(self, logits, labels, is_logits=True, is_binary=False, accuracies=None, confidences=None):
        if confidences is None:
            if is_logits:
                softmaxes = F.softmax(logits, dim=1)
                if is_binary:
                    confidences = softmaxes[:, 0]
                else:
                    confidences, predictions = torch.max(softmaxes, 1)
            else:
                confidences, predictions = torch.max(logits, 1)
        if accuracies is None:
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

def class_temperature_scale(logits, csece_temperature):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits
    return logits / csece_temperature

def temperature_scale(logits, temperature):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits
    return logits / temperature

def test_classification_net_logits(logits, labels, is_logits=True, is_binary=False, accuracies=None, predictions=None):
    '''
    This function reports classification accuracy and confusion matrix given logits and labels
    from a model.
    '''
    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    if is_logits:
        softmax = F.softmax(logits, dim=1)
        if is_binary:
            confidence_vals = softmax[:, 0]
        else:
            confidence_vals, predictions = torch.max(softmax, 1)
    else:
        confidence_vals, predictions = torch.max(logits, dim=1)
    labels_list.extend(labels.cpu().numpy().tolist())
    predictions_list.extend(predictions.cpu().numpy().tolist())
    confidence_vals_list.extend(confidence_vals.cpu().numpy().tolist())
    if accuracies is None:
        accuracy = accuracy_score(labels_list, predictions_list)
    else:
        accuracy = accuracies.float().mean()
    return confusion_matrix(labels_list, predictions_list), accuracy, labels_list,\
        predictions_list, confidence_vals_list