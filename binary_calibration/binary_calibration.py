import pickle
import torch
from torch.nn import functional as F

from binary_utils import test_classification_net_logits, class_temperature_scale, temperature_scale, ECELoss


def set_binary_calibration(logits, labels, acc_check=False, num_bins=10):
    
    """
    Tune binary tempearature for the model (using the validation set) with cross-validation on ECE
    """
    # Calculate ECE before temperature scaling
    ece_criterion = ECELoss(n_bins=num_bins).cuda()
    before_temperature_ece = ece_criterion(logits, labels).item()
    print('Before temperature - ECE: %.3f' % (before_temperature_ece))

    ece_val = 10 ** 7
    T_opt_ece = 1.0
    T = 0.1
    for i in range(100):
        temperature = T
        after_temperature_ece = ece_criterion(temperature_scale(logits, temperature), labels).item()
        
        if ece_val > after_temperature_ece:
            T_opt_ece = T
            ece_val = after_temperature_ece
        T += 0.1

    init_temp = T_opt_ece

    # Calculate ECE after temperature scaling
    after_temperature_ece = ece_criterion(temperature_scale(logits, init_temp), labels).item()
    print('Optimal temperature: %.3f' % init_temp)
    print('After temperature - ECE: %.3f' % (after_temperature_ece))
    
    """
    Find tempearature vector for the model (using the validation set) with cross-validation on ECE
    """
    
    # Calculate ECE before temperature scaling
    if acc_check:
        _, accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    print('Before temperature - ECE: {0:.3f}'.format(before_temperature_ece))

    T_csece = init_temp*torch.ones(logits.size()[1]).cuda()
    csece_temperature = T_csece.clone()
    
    # Determine model accuracy after single TS
    if acc_check:
        _, temp_accuracy, _, _, _ = test_classification_net_logits(class_temperature_scale(logits, csece_temperature), labels)
        if temp_accuracy >= accuracy:
            accuracy = temp_accuracy

    ece_val = 10 ** 7
    csece_val = 10 ** 7
    for pred_label in range(logits.size()[1]):
        probs = F.softmax(logits, dim=-1)
        confidences, preds = torch.max(probs, dim=-1)
        if not (preds == pred_label).any():
            continue
        true_labels_label = labels[preds == pred_label]
        pred_label_vec = preds[preds == pred_label]
        accuracies_label = pred_label_vec.eq(true_labels_label)
        probs_label = probs[preds == pred_label]
        binary_probs = torch.zeros((probs_label.shape[0], 2), device=probs_label.device)
        binary_probs[:, 0] = confidences[preds == pred_label]
        binary_probs[:, 1] = 1.0 - confidences[preds == pred_label]
        binary_logits = torch.log(binary_probs)
        current_labels = pred_label * torch.ones(binary_logits.shape[0], device=binary_logits.device)
        T = 0.1
        best_temp = init_temp
        for i in range(100):
            after_temperature_ece = ece_criterion(binary_logits / T, current_labels, is_binary=True, accuracies=accuracies_label).item()
            
            if acc_check:
                _, temp_accuracy, _, _, _ = test_classification_net_logits(binary_logits / T, current_labels, is_binary=True, accuracies=accuracies_label, predictions=pred_label_vec)

            if acc_check:  # Change temperature only if accuracy preserved
                if csece_val > after_temperature_ece and temp_accuracy >= accuracy:
                    best_temp = T
                    csece_val = after_temperature_ece
                    accuracy = temp_accuracy
            else:
                if csece_val > after_temperature_ece:
                    best_temp = T
                    csece_val = after_temperature_ece
            T += 0.1
        T_csece[pred_label] = best_temp
        
    csece_temperature = T_csece

    return csece_temperature, init_temp

def infer_binary_calibration(logits, labels, csece_temperature):
        """
        Perform temperature scaling on logits
        """
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        new_confidences = confidences.clone()
        accuracies = predictions.eq(labels)
        # confidences[confidences > 0.99999] = 0.99999
        n_classes = logits.shape[1]
        for pred_label in range(n_classes):
            #init_temp_value = T_csece[label].item()
            probs = F.softmax(logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
            if not (preds == pred_label).any():
                continue
            probs_label = probs[preds == pred_label]
            binary_probs = torch.zeros((probs_label.shape[0], 2), device=probs_label.device)
            binary_probs[:, 0] = confidences[preds == pred_label]
            binary_probs[:, 1] = 1.0 - confidences[preds == pred_label]
            binary_logits = torch.log(binary_probs)
            new_confidences[preds == pred_label] = F.softmax(binary_logits / csece_temperature[pred_label], dim=1)[:, 0]
                            
        return new_confidences, accuracies

def main():
    # Setting additional parameters
    torch.manual_seed(1)
    
    # Example data, just need logits + labels from your model and data
    model_name = 'resnet50'
    logits_path = 'binary_calibration/data/CXR14'
    with open(logits_path + '/logits/' + model_name + '/test_labels.pickle', 'rb') as handle:
        labels_test = pickle.load(handle)
    with open(logits_path + '/logits/' + model_name + '/test_logits.pickle', 'rb') as handle:
        logits_test = pickle.load(handle)
    with open(logits_path + '/logits/' + model_name + '/val_labels.pickle', 'rb') as handle:
        labels_val = pickle.load(handle)
    with open(logits_path + '/logits/' + model_name + '/val_logits.pickle', 'rb') as handle:
        logits_val = pickle.load(handle)
            
    # Calculate pre-calibration ECE
    ece_criterion = ECELoss(n_bins=10).cuda()
   
    p_ece = ece_criterion(logits_test, labels_test).item()
    _, p_acc, _, _, _ = test_classification_net_logits(logits_test, labels_test)
    
    # Printing the required evaluation metrics
    print('Pre-scaling test ECE: ' + str(p_ece))
    print('Pre-scaling test accuracy: ' + str(p_acc))
        
    per_class_temperature, single_temperature = set_binary_calibration(logits_val, labels_val, acc_check=True)        
    new_confidences, new_accuracies = infer_binary_calibration(logits_test, labels_test, per_class_temperature)
    
    ece = ece_criterion(logits_test, labels_test, accuracies=new_accuracies, confidences=new_confidences).item()
    ece_single = ece_criterion(temperature_scale(logits_test, single_temperature), labels_test).item()
    
    print ('Post-scaling ECE (Binary scaling): ' + str(ece))
    print ('Post-scaling ECE (Single temp scaling): ' + str(ece_single))
    print ('Post-scaling accuracy: ' + str(p_acc))
    print ('Pre-scaling ECE: ' + str(p_ece))
        
if __name__ == "__main__":
    main()
    