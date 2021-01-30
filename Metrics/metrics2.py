import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)

def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin
  

def ECE(conf, pred, true, bin_size = 0.1):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        ece: expected calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)        
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece

# Calibration error scores in the form of loss metrics
class ECE2(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, bin_size=0.1):
        """
        Expected Calibration Error

        Args:
            conf (numpy.ndarray): list of confidences
            pred (numpy.ndarray): list of predictions
            true (numpy.ndarray): list of true labels
            bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

        Returns:
            ece: expected calibration error
        """
        super(ECE2, self).__init__()
        self.upper_bounds = torch.linspace(bin_size, 1+bin_size, bin_size)  # Get bounds of bins

    def compute_acc_bin(self, conf_thresh_lower, conf_thresh_upper, conf, pred, true):
        """
        # Computes accuracy and average confidence for bin

        Args:
            conf_thresh_lower (float): Lower Threshold of confidence interval
            conf_thresh_upper (float): Upper Threshold of confidence interval
            conf (numpy.ndarray): list of confidences
            pred (numpy.ndarray): list of predictions
            true (numpy.ndarray): list of true labels

        Returns:
            (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
        """
        filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
        if len(filtered_tuples) < 1:
            return 0,0,0
        else:
            correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
            len_bin = len(filtered_tuples)  # How many elements falls into given bin
            avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
            accuracy = float(correct)/len_bin  # accuracy of BIN
            return accuracy, avg_conf, len_bin
    
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        conf, pred = torch.max(softmaxes, 1)
        
        n = len(conf)
        ece = torch.zeros(1, device=conf.device)  # Starting error
        
        for conf_thresh in self.upper_bounds:  # Go through bounds and find accuracies and confidences
            acc, avg_conf, len_bin = self.compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)        
            ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE

        return ece

def test_classification_net_logits2(logits, labels):
    '''
    This function reports classification accuracy and confusion matrix given logits and labels
    from a model.
    '''
    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    softmaxs = softmax(logits)
    confidence_vals = np.max(softmaxs, axis=1)
    predictions = np.argmax(softmaxs, axis=1)
    labels_list.extend(labels.tolist())
    predictions_list.extend(predictions.tolist())
    confidence_vals_list.extend(confidence_vals.tolist())
    accuracy = accuracy_score(labels_list, predictions_list)
    return confusion_matrix(labels_list, predictions_list), accuracy, labels_list,\
        predictions_list, confidence_vals_list
