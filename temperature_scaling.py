'''
Code to perform temperature scaling. Adapted from https://github.com/gpleiss/temperature_scaling
'''
from numpy.core.numeric import cross
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import math

from Metrics.metrics import test_classification_net_logits
from Metrics.metrics import ECELoss, ClassECELoss, posnegECELoss, estECELoss
from Metrics.metrics2 import ECE, softmax, test_classification_net_logits2

torch.set_printoptions(precision=10)

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, log=True, const_temp=False, bins_temp=False, n_bins=15, iters=1, dists=False, grid=True):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = 1.0
        self.log = log
        self.const_temp = const_temp
        self.ece_list = []
        self.ece = 0.0
        self.bins_temp = bins_temp
        self.dists = dists
        self.n_bins = n_bins
        self.iters = iters  # Number of maximum iterations
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1).unsqueeze(0).repeat((iters, 1)).numpy()
        self.best_iter = 0  # Best iteration for scaling
        self.temps_iters = torch.ones(iters).cuda()  # Temperatures fot iter single TS
        self.grid = grid  # Applying grid search to find optimal weight


    def forward(self, input, labels, const_temp=False, bins_temp=False):
        logits = self.model(input)
        if self.const_temp or const_temp:
            return self.temperature_scale(logits)
            #return self.iter_temperature_scale(logits)
        elif bins_temp:
            return self.bins_temperature_scale_test(logits, labels.cuda(), n_bins=self.n_bins)
        elif self.dists:
            return self.single_dists_test(logits)
        else:
            return self.class_temperature_scale(logits)


    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        return logits / self.temperature
    
    def iter_temperature_scale(self, logits):
        """
        Perform iterative temperature scaling on logits
        """
        scaled_logits = logits.clone()
        for i in range(self.iters):
            scaled_logits = scaled_logits / self.temps_iters[i]
        # Expand temperature to match the size of logits
        return scaled_logits
    
    def class_temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        return logits / self.csece_temperature
    
    def bins_temperature_scale_test(self, logits, labels, n_bins=15):
        """
        Perform temperature scaling on logits
        """
        ece_criterion = ECELoss(n_bins=n_bins).cuda()
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        # confidences[confidences == 1] = 0.999999
        scaled_logits = logits.clone()
        ece_list = []
        for i in range(self.best_iter + 1):
            bin = 0
            bin_lowers = self.bin_boundaries[i][:-1]
            bin_uppers = self.bin_boundaries[i][1:]
            print('\n')
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                accuracies_temp = accuracies[in_bin]
                # accuracy_in_bin = accuracies_temp.float().mean().item()
                accuracy_in_bin = min(accuracies_temp.float().mean().item(), 0.99)
                accuracy_in_bin = max(accuracy_in_bin, 0.01)
                if any(in_bin):
                    scaled_logits[in_bin] = scaled_logits[in_bin] / self.bins_T[bin, i]
                    softmaxes_temp = F.softmax(scaled_logits[in_bin], dim=1)
                    confidences_temp, _ = torch.max(softmaxes_temp, 1)
                    avg_confidence_in_bin = confidences_temp.mean()
                    after_temperature = torch.abs(accuracy_in_bin - avg_confidence_in_bin)
                    samples = scaled_logits[in_bin].shape[0]
                    print('ece in bin ', bin + 1, ' :', (prop_in_bin * after_temperature).item(),
                          ', number of samples: ', samples)
                    print('accuracy in bin ', bin + 1, ': ', accuracy_in_bin)
                bin += 1

            ece_list.append(ece_criterion(scaled_logits, labels).item())
            softmaxes = F.softmax(scaled_logits, dim=1)
            confidences, _ = torch.max(softmaxes, 1)

        print(ece_list)
        print('Number of iters: {}'.format(self.best_iter + 1))

        return scaled_logits

    def bins_dists_scale_test(self, logits, labels, n_bins=15):
        """
        Perform temperature scaling on logits
        """
        ece_criterion = ECELoss(n_bins=n_bins).cuda()
        new_softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(new_softmaxes, 1)
        accuracies = predictions.eq(labels)
        # confidences[confidences > 0.99999] = 0.99999
        ece_list = []
        n_classes = logits.shape[1]
        for i in range(self.best_iter + 1):
            bin = 0
            bin_lowers = self.bin_boundaries[i][:-1]
            bin_uppers = self.bin_boundaries[i][1:]
            print('\n')
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                accuracies_temp = accuracies[in_bin]
                origin_accuracy_in_bin = accuracies_temp.float().mean().item()
                accuracy_in_bin = min(accuracies_temp.float().mean().item(), 0.99)
                accuracy_in_bin = max(accuracy_in_bin, 0.01)
                if any(in_bin):
                    weight = self.bins_T[bin, i]
                    new_softmaxes[in_bin] = weight * new_softmaxes[in_bin] + (1 - weight) * 1 / n_classes
                    confidences_temp, _ = torch.max(new_softmaxes[in_bin], 1)
                    avg_confidence_in_bin = confidences_temp.mean()
                    after_temperature = torch.abs(accuracy_in_bin - avg_confidence_in_bin)
                    samples = new_softmaxes[in_bin].shape[0]
                    print('ece in bin ', bin + 1, ' :', (prop_in_bin * after_temperature).item(),
                          ', number of samples: ', samples)
                    print('accuracy in bin ', bin + 1, ': ', origin_accuracy_in_bin)
                bin += 1

            new_softmaxes /= torch.sum(new_softmaxes, 1, keepdim=True)
            ece_list.append(ece_criterion(new_softmaxes, labels, is_logits=False).item())
            confidences, _ = torch.max(new_softmaxes, 1)

        print(ece_list)
        print('Number of iters: {}'.format(self.best_iter + 1))

        return new_softmaxes
    
    def single_dists_test(self, logits):
        """
        Perform temperature scaling on logits
        """
        softmaxes = F.softmax(logits, 1)
        # confidence, predictions = torch.max(softmaxes, 1, keepdim=True)
        # sorted_conf, sorted_pred = torch.sort(softmaxes, 1, True)
        n_classes = logits.shape[1]
        
        # softmaxes = softmaxes - self.weight * dists
        softmaxes = self.weight * softmaxes + (1 - self.weight) * 1 / n_classes
        # conf = sorted_conf[:, 0]
        # second_conf = sorted_conf[:, 1]
        # new_values = (1 - self.weight) * conf + self.weight * second_conf
        # softmaxes.scatter_(1, predictions, new_values.unsqueeze(-1))
        # softmaxes = self.weight * softmaxes
        softmaxes /= torch.sum(softmaxes, 1, keepdim=True)
                                    
        return softmaxes
    
    def bins_temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """     
        # Expand temperature to match the size of logits
        return logits / torch.unsqueeze(self.bece_temperature, -1)
    

    def set_temperature(self, valid_loader, cross_validate='ece', init_temp=2.5, acc_check=False):
        """
        Tune the tempearature of the model (using the validation set) with cross-validation on ECE or NLL
        """
        if self.const_temp:
            self.cuda()
            self.model.eval()
            nll_criterion = nn.CrossEntropyLoss().cuda()
            ece_criterion = ECELoss().cuda()

            # First: collect all the logits and labels for the validation set
            logits_list = []
            labels_list = []
            with torch.no_grad():
                for input, label in valid_loader:
                    input = input.cuda()
                    logits = self.model(input)
                    logits_list.append(logits)
                    labels_list.append(label)
                logits = torch.cat(logits_list).cuda()
                labels = torch.cat(labels_list).cuda()

            # Calculate NLL and ECE before temperature scaling
            before_temperature_nll = nll_criterion(logits, labels).item()
            before_temperature_ece = ece_criterion(logits, labels).item()
            if self.log:
                print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

            nll_val = 10 ** 7
            ece_val = 10 ** 7
            T_opt_nll = 1.0
            T_opt_ece = 1.0
            T = 0.1
            for i in range(100):
                self.temperature = T
                self.cuda()
                after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
                after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
                if nll_val > after_temperature_nll:
                    T_opt_nll = T
                    nll_val = after_temperature_nll

                if ece_val > after_temperature_ece:
                    T_opt_ece = T
                    ece_val = after_temperature_ece
                T += 0.1

            if cross_validate == 'ece':
                self.temperature = T_opt_ece
            else:
                self.temperature = T_opt_nll
            self.cuda()

            # Calculate NLL and ECE after temperature scaling
            after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            if self.log:
                print('Optimal temperature: %.3f' % self.temperature)
                print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        
        else:
            self.cuda()
            self.model.eval()
            nll_criterion = nn.CrossEntropyLoss().cuda()
            ece_criterion = ECELoss().cuda()
            csece_criterion = ClassECELoss().cuda()
            posneg_csece_criterion = posnegECELoss().cuda()

            # First: collect all the logits and labels for the validation set
            logits_list = []
            labels_list = []
            with torch.no_grad():
                for input, label in valid_loader:
                    input = input.cuda()
                    logits = self.model(input)
                    logits_list.append(logits)
                    labels_list.append(label)
                logits = torch.cat(logits_list).cuda()
                labels = torch.cat(labels_list).cuda()

            before_temperature_ece = ece_criterion(logits, labels).item()
            if self.log:
                print('Before temperature - ECE: %.3f' % (before_temperature_ece))

            ece_val = 10 ** 7
            T_opt_ece = 1.0
            T = 0.1
            for i in range(100):
                self.temperature = T
                self.cuda()
                after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
                if ece_val > after_temperature_ece:
                    T_opt_ece = T
                    ece_val = after_temperature_ece
                T += 0.1

            init_temp = T_opt_ece
            self.temperature = T_opt_ece
            
            # Calculate NLL and ECE after temperature scaling
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            if self.log:
                print('Optimal temperature: %.3f' % init_temp)
                print('After temperature - ECE: %.3f' % (after_temperature_ece))

            """
            Find tempearature vector for the model (using the validation set) with cross-validation on ECE
            """
            T_opt_nll = 1.0
            T_opt_ece = 1.0
            T_opt_csece = init_temp*torch.ones(logits.size()[1]).cuda()
            T_csece = init_temp*torch.ones(logits.size()[1]).cuda()
            self.csece_temperature = T_csece
            self.ece_list.append(ece_criterion(self.class_temperature_scale(logits), labels).item())
            _, accuracy, _, _, _ = test_classification_net_logits(logits, labels)
            if acc_check:
                _, temp_accuracy, _, _, _ = test_classification_net_logits(self.class_temperature_scale(logits), labels)
                if temp_accuracy >= accuracy:
                    accuracy = temp_accuracy
            
            steps_limit = 0.2
            temp_steps = torch.linspace(-steps_limit, steps_limit, int((2 * steps_limit) / 0.1 + 1))
            converged = False
            prev_temperatures = self.csece_temperature.clone()
            nll_val = 10 ** 7
            ece_val = 10 ** 7
            csece_val = 10 ** 7
                 
            #for iter in range(self.iters):
            while not converged:
                for label in range(logits.size()[1]):
                    init_temp_value = T_csece[label].item()
                    #T = 0.1
                    #for i in range(100):
                    for step in temp_steps:
                        #T_csece[label] = T
                        T_csece[label] = init_temp_value + step
                        self.csece_temperature = T_csece
                        #self.temperature = T
                        self.cuda()
                        #after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
                        after_temperature_ece = ece_criterion(self.class_temperature_scale(logits), labels).item()
                        #after_temperature_ece_reg = ece_criterion(self.temperature_scale(logits), labels).item()
                        if acc_check:
                            _, temp_accuracy, _, _, _ = test_classification_net_logits(self.class_temperature_scale(logits), labels)

                        if acc_check:
                            if csece_val > after_temperature_ece and temp_accuracy >= accuracy:
                                T_opt_csece[label] = T
                                csece_val = after_temperature_ece
                                accuracy = temp_accuracy
                        else:
                            if csece_val > after_temperature_ece:
                                #T_opt_csece[label] = T
                                T_opt_csece[label] = init_temp_value + step
                                csece_val = after_temperature_ece
                        #T += 0.1
                    T_csece[label] = T_opt_csece[label]
                self.csece_temperature = T_opt_csece
                self.ece_list.append(ece_criterion(self.class_temperature_scale(logits), labels).item())
                converged = torch.all(self.csece_temperature.eq(prev_temperatures))
                prev_temperatures = self.csece_temperature.clone()

            self.csece_temperature = T_opt_csece
            self.cuda()
        return self


    def get_temperature(self):
        if self.const_temp:
            return self.temperature
        elif self.bins_temp:
            return self.temperature, self.bins_T
        elif self.dists:
            return self.weight
        else:
            return self.temperature, self.csece_temperature
        
    def set_bins_temperature(self, valid_loader, cross_validate='ece', init_temp=2.5, acc_check=False, n_bins=15):
        """
        Tune the tempearature of the model (using the validation set) with cross-validation on ECE or NLL
        """
        self.cuda()
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if self.log:
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
            
        eps = 1e-6
        ece_val = 10 ** 7
        T_opt_ece = 1.0
        T = 0.1
        for i in range(100):
            self.temperature = T
            self.cuda()
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        init_temp = T_opt_ece
        self.temperature = T_opt_ece
        
        # Calculate NLL and ECE after temperature scaling
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        if self.log:
            print('Optimal temperature: %.3f' % init_temp)
            print('After temperature - ECE: %.3f' % (after_temperature_ece))

        T_opt_bece = init_temp*torch.ones(logits.shape[0]).cuda()
        T_bece = init_temp*torch.ones(logits.shape[0]).cuda()
        self.bins_T = init_temp*torch.ones(n_bins).cuda()
        #bins_T_opt = init_temp*torch.ones(n_bins).cuda()
        self.bece_temperature = T_bece
        
        self.ece_list.append(ece_criterion(self.temperature_scale(logits), labels).item())
        _, accuracy, _, _, _ = test_classification_net_logits(logits, labels)
        if acc_check:
            _, temp_accuracy, _, _, _ = test_classification_net_logits(self.temperature_scale(logits), labels)
            if temp_accuracy >= accuracy:
                accuracy = temp_accuracy
        
        softmaxes = F.softmax(logits, dim=1)
        confidences, _ = torch.max(softmaxes, 1)
        
        steps_limit = 0.2
        temp_steps = torch.linspace(-steps_limit, steps_limit, int((2 * steps_limit) / 0.1 + 1)).cuda()
        converged = False
        prev_temperatures = self.bece_temperature.clone()
        #prev_temperatures = self.bins_T.clone()
        bece_val = 10 ** 7
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
                
        self.iters = 0
        while not converged:
            self.iters += 1
            bin = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                #prop_in_bin = in_bin.float().mean()
                if any(in_bin):
                    init_temp_value = T_bece[in_bin][0].item()
                    #init_temp_value = self.bins_T[bin].item()
                    for step in temp_steps:
                        T_bece[in_bin] = init_temp_value + step
                        self.bece_temperature = T_bece
                        #self.bins_T[bin] = init_temp_value + step
                        self.cuda()
                        after_temperature_ece = ece_criterion(self.bins_temperature_scale(logits), labels).item()
                        if acc_check:
                            _, temp_accuracy, _, _, _ = test_classification_net_logits(self.bins_temperature_scale(logits), labels)

                        if acc_check:
                            if bece_val > after_temperature_ece + eps and temp_accuracy >= accuracy:
                                T_opt_bece[in_bin] = init_temp_value + step
                                #bins_T_opt[bin] = init_temp_value + step
                                bece_val = after_temperature_ece
                                accuracy = temp_accuracy
                        else:
                            if bece_val > after_temperature_ece + eps:
                                T_opt_bece[in_bin] = init_temp_value + step
                                #bins_T_opt[bin] = init_temp_value + step
                                bece_val = after_temperature_ece
                    T_bece[in_bin] = T_opt_bece[in_bin]
                    #self.bins_T[bin] = bins_T_opt[bin]
                    self.bins_T[bin] = T_bece[in_bin][0].item()
                bin += 1
            self.bece_temperature = T_opt_bece
            #self.bins_T = bins_T_opt
            self.ece_list.append(ece_criterion(self.bins_temperature_scale(logits), labels).item())
            converged = torch.all(self.bece_temperature.eq(prev_temperatures))
            prev_temperatures = self.bece_temperature.clone()
            
        self.bece_temperature = T_opt_bece
        #self.bins_T = bins_T_opt
        self.cuda()
        
        return self
    
    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.n_bins + 1),
                     np.arange(npt),
                     np.sort(x))
    
    def set_bins_temperature2(self, valid_loader, cross_validate='ece', init_temp=2.5, acc_check=False, top_temp=10):
        """
        Tune the tempearature of the model (using the validation set) with cross-validation on ECE or NLL
        """
        self.cuda()
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if self.log:
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
            
        n_bins = self.n_bins
        eps = 1e-6
        ece_val = 10 ** 7
        T_opt_ece = 1.0
        T = 0.1
        for i in range(100):
            self.temperature = T
            self.cuda()
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        init_temp = T_opt_ece
        self.temperature = T_opt_ece
        
        # Calculate NLL and ECE after temperature scaling
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        if self.log:
            print('Optimal temperature: %.3f' % init_temp)
            print('After temperature - ECE: %.3f' % (after_temperature_ece))

        init_temp = 1
        T_opt_bece = init_temp*torch.ones(logits.shape[0]).cuda()
        T_bece = init_temp*torch.ones(logits.shape[0]).cuda()
        self.bins_T = init_temp*torch.ones((n_bins, self.iters)).cuda()
        self.bece_temperature = T_bece
        
        self.ece_list.append(ece_criterion(self.temperature_scale(logits), labels).item())
                
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
                        
        for i in range(self.iters):
            ece_in_iter = 0
            print('iter num ', i+1)
            bin = 0
            few_examples = dict()
            n, self.bin_boundaries[i] = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
            bin_lowers = self.bin_boundaries[i][:-1]
            bin_uppers = self.bin_boundaries[i][1:]
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                bece_val = 10 ** 7
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if confidences[in_bin].shape[0] < 20:
                    samples = T_bece[in_bin].shape[0]
                    print('number of samples in bin {0}: {1}'.format(bin + 1, samples))
                    few_examples[bin] = samples
                    bin += 1
                    continue
                if any(in_bin):
                    init_temp_value = T_bece[in_bin][0].item()
                    T = 0.1
                    accuracies_temp = accuracies[in_bin]
                    accuracy_in_bin = min(accuracies_temp.float().mean().item(), 0.99)
                    accuracy_in_bin = max(accuracy_in_bin, 0.01)
                    for t in range(100):
                        T_bece[in_bin] = T
                        self.bece_temperature = T_bece
                        
                        softmaxes_temp = F.softmax(logits[in_bin] / torch.unsqueeze(T_bece[in_bin], -1), dim=1)
                        confidences_temp, _ = torch.max(softmaxes_temp, 1)
                        avg_confidence_in_bin = confidences_temp.mean()
                        after_temperature = torch.abs(accuracy_in_bin - avg_confidence_in_bin)
                        
                        if bece_val > after_temperature + eps:
                            T_opt_bece[in_bin] = T
                            bece_val = after_temperature
                            #print('conf-acc: ', (avg_confidence_in_bin - accuracy_in_bin).item())
                            #print('temp: ', T)
                        T += 0.1
                        
                    T_bece[in_bin] = T_opt_bece[in_bin]
                    self.bins_T[bin, i] = T_opt_bece[in_bin][0].item()
                    
                    samples = T_bece[in_bin].shape[0]
                    ece_in_iter += prop_in_bin * bece_val
                    print('ece in bin ', bin+1, ' :', (prop_in_bin * bece_val).item(), ', number of samples: ', samples)
                bin += 1

            for bin in few_examples:
                #bins_T[bin, i] = self.temperature

                if bin > 0 and bin < n_bins - 1:
                    lower_bin = bin - 1
                    upper_bin = bin + 1
                    while lower_bin in few_examples and lower_bin - 1 >= 0:
                        lower_bin -= 1
                    while upper_bin in few_examples and upper_bin + 1 <= n_bins - 1:
                        upper_bin += 1
                    if upper_bin == n_bins - 1:
                        self.bins_T[bin, i] = self.bins_T[lower_bin, i]
                    else:
                        avg_temp = (self.bins_T[lower_bin, i] + self.bins_T[upper_bin, i]) / 2  # Mean temperature of neighbors
                        self.bins_T[bin, i] = avg_temp
                elif bin == 0:
                    upper_bin = bin + 1
                    while upper_bin in few_examples and upper_bin + 1 <= n_bins - 1:
                        upper_bin += 1
                    self.bins_T[bin, i] = self.bins_T[upper_bin, i]
                else:
                    lower_bin = bin - 1
                    while lower_bin in few_examples and lower_bin - 1 >= 0:
                        lower_bin -= 1
                    self.bins_T[bin, i] = self.bins_T[lower_bin, i]
            
            self.bece_temperature = T_opt_bece
            current_ece = ece_criterion(self.bins_temperature_scale(logits), labels).item()
            print('ece in iter ', i + 1, ' :', current_ece)
            if i > 0 and current_ece < self.ece_list[self.best_iter]:
                self.best_iter = i
            if abs(self.ece_list[-1] - current_ece) > eps:
                self.ece_list.append(current_ece)
            else:
                self.iters = i + 1
                break
            
            logits = logits / torch.unsqueeze(self.bece_temperature, -1)
            softmaxes = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmaxes, 1)
            
        self.bece_temperature = T_opt_bece

        return self
    
    def set_bins_dists(self, valid_loader, cross_validate='ece', init_temp=2.5, acc_check=False, top_temp=10):
            """
            Tune the tempearature of the model (using the validation set) with cross-validation on ECE or NLL
            """
            self.cuda()
            self.model.eval()
            nll_criterion = nn.CrossEntropyLoss().cuda()
            ece_criterion = ECELoss().cuda()

            # First: collect all the logits and labels for the validation set
            logits_list = []
            labels_list = []
            with torch.no_grad():
                for input, label in valid_loader:
                    input = input.cuda()
                    logits = self.model(input)
                    logits_list.append(logits)
                    labels_list.append(label)
                logits = torch.cat(logits_list).cuda()
                labels = torch.cat(labels_list).cuda()

            # Calculate NLL and ECE before temperature scaling
            before_temperature_nll = nll_criterion(logits, labels).item()
            before_temperature_ece = ece_criterion(logits, labels).item()
            if self.log:
                print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
                
            n_bins = self.n_bins
            eps = 1e-6
            ece_val = 10 ** 7
            T_opt_ece = 1.0
            T = 0.1
            for i in range(100):
                self.temperature = T
                self.cuda()
                after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
                if ece_val > after_temperature_ece:
                    T_opt_ece = T
                    ece_val = after_temperature_ece
                T += 0.1

            init_temp = T_opt_ece
            self.temperature = T_opt_ece
            
            # Calculate NLL and ECE after temperature scaling
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            if self.log:
                print('Optimal temperature: %.3f' % init_temp)
                print('After temperature - ECE: %.3f' % (after_temperature_ece))

            init_temp = 1
            T_opt_bece = init_temp*torch.ones(logits.shape[0]).cuda()
            T_bece = init_temp*torch.ones(logits.shape[0]).cuda()
            self.bins_T = init_temp*torch.ones((n_bins, self.iters)).cuda()
            self.bece_temperature = T_bece
            
            self.ece_list.append(ece_criterion(self.temperature_scale(logits), labels).item())
                    
            softmaxes = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmaxes, 1)
            # confidences[confidences > 0.99999] = 0.99999
            accuracies = predictions.eq(labels)
            n_classes = logits.shape[1]
                            
            for i in range(self.iters):
                ece_in_iter = 0
                print('iter num ', i+1)
                bin = 0
                few_examples = dict()
                n, self.bin_boundaries[i] = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
                bin_lowers = self.bin_boundaries[i][:-1]
                bin_uppers = self.bin_boundaries[i][1:]
                
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    bece_val = 10 ** 7
                    in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                    prop_in_bin = in_bin.float().mean()
                    if confidences[in_bin].shape[0] < 20:
                        samples = T_bece[in_bin].shape[0]
                        print('number of samples in bin {0}: {1}'.format(bin + 1, samples))
                        few_examples[bin] = samples
                        bin += 1
                        continue
                    if any(in_bin):
                        init_temp_value = T_bece[in_bin][0].item()
                        origin_avg_confidence_in_bin = confidences[in_bin].mean()
                        origin_accuracy_in_bin = accuracies[in_bin].float().mean().item()
                        accuracy_in_bin = min(origin_accuracy_in_bin, 0.99)
                        accuracy_in_bin = max(accuracy_in_bin, 0.01)
                        
                        if self.grid:
                            T = 0.1
                            for t in range(100):
                                temp_softmaxes = softmaxes.clone()
                                T_bece[in_bin] = T
                                temp_softmaxes[in_bin] = T * temp_softmaxes[in_bin] + (1 - T) * 1 / n_classes
                                temp_softmaxes /= torch.sum(temp_softmaxes, 1, keepdim=True)
                                confidences_temp, _ = torch.max(temp_softmaxes[in_bin], 1)
                                self.bece_temperature = T_bece
                                
                                avg_confidence_in_bin = confidences_temp.mean()
                                after_temperature = torch.abs(accuracy_in_bin - avg_confidence_in_bin)
                                
                                if bece_val > after_temperature + eps:
                                    T_opt_bece[in_bin] = T
                                    bece_val = after_temperature
                                    #print('conf-acc: ', (avg_confidence_in_bin - accuracy_in_bin).item())
                                    #print('temp: ', T)
                                T += 0.1
                        
                        else:    
                            weight_i = (origin_accuracy_in_bin - 1/n_classes) / (origin_avg_confidence_in_bin - 1/n_classes)
                            weight_i = max(0, min(1, weight_i.item()))
                            T_opt_bece[in_bin] = weight_i
                        
                            # Compute bin's updated ECE
                            temp_softmaxes = softmaxes.clone()
                            temp_softmaxes[in_bin] = weight_i * temp_softmaxes[in_bin] + (1 - weight_i) * 1 / n_classes
                            temp_softmaxes /= torch.sum(temp_softmaxes, 1, keepdim=True)
                            confidences_temp, _ = torch.max(temp_softmaxes[in_bin], 1)
                            avg_confidence_in_bin = confidences_temp.mean()
                            bece_val = torch.abs(origin_accuracy_in_bin - avg_confidence_in_bin)
                        
                        T_bece[in_bin] = T_opt_bece[in_bin]
                        self.bins_T[bin, i] = T_opt_bece[in_bin][0].item()
                        
                        samples = T_bece[in_bin].shape[0]
                        ece_in_iter += prop_in_bin * bece_val
                        print('ece in bin ', bin+1, ' :', (prop_in_bin * bece_val).item(), ', number of samples: ', samples)
                    bin += 1

                for bin in few_examples:
                    #bins_T[bin, i] = self.temperature

                    if bin > 0 and bin < n_bins - 1:
                        lower_bin = bin - 1
                        upper_bin = bin + 1
                        while lower_bin in few_examples and lower_bin - 1 >= 0:
                            lower_bin -= 1
                        while upper_bin in few_examples and upper_bin + 1 <= n_bins - 1:
                            upper_bin += 1
                        if upper_bin == n_bins - 1:
                            self.bins_T[bin, i] = self.bins_T[lower_bin, i]
                        else:
                            avg_temp = (self.bins_T[lower_bin, i] + self.bins_T[upper_bin, i]) / 2  # Mean temperature of neighbors
                            self.bins_T[bin, i] = avg_temp
                    elif bin == 0:
                        upper_bin = bin + 1
                        while upper_bin in few_examples and upper_bin + 1 <= n_bins - 1:
                            upper_bin += 1
                        self.bins_T[bin, i] = self.bins_T[upper_bin, i]
                    else:
                        lower_bin = bin - 1
                        while lower_bin in few_examples and lower_bin - 1 >= 0:
                            lower_bin -= 1
                        self.bins_T[bin, i] = self.bins_T[lower_bin, i]
                
                self.bece_temperature = T_opt_bece
                current_ece = ece_criterion(self.bins_temperature_scale(logits), labels).item()
                print('ece in iter ', i + 1, ' :', current_ece)
                if i > 0 and current_ece < self.ece_list[self.best_iter]:
                    self.best_iter = i
                if abs(self.ece_list[-1] - current_ece) > eps:
                    self.ece_list.append(current_ece)
                else:
                    self.iters = i + 1
                    break
                
                weight = self.bece_temperature.unsqueeze(-1).repeat(1, logits.shape[1])
                softmaxes = weight * softmaxes + (1 - weight) * 1 / n_classes
                confidences, predictions = torch.max(softmaxes, 1)
                
            self.bece_temperature = T_opt_bece

            return self

    def set_single_dists(self, val_loader, log=True):
        """
        Tune the tempearature of the model (using the validation set) with cross-validation on ECE without bins
        """
        ece_criterion = ECELoss().cuda()
        
        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in val_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate ECE before temperature scaling
        before_temperature_ece = ece_criterion(logits, labels).item()
        if log:
            print('Before temperature - ECE: %.3f' % before_temperature_ece)
            
        ece_val = 10 ** 7
        T_opt_ece = 1.0
        T = 0.1
        for i in range(100):
            self.temperature = T
            self.cuda()
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        init_temp = T_opt_ece
        self.temperature = T_opt_ece
        
        # Calculate NLL and ECE after temperature scaling
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        if self.log:
            print('Optimal temperature: %.3f' % init_temp)
            print('After temperature - ECE: %.3f' % (after_temperature_ece))
    
        softmaxes = F.softmax(logits, 1)
        n_classes = logits.shape[1]
        eps = 1e-6
        ece_val = 10 ** 7
        T_opt_ece = 1.0

        if not self.grid:
            confidences, predictions = torch.max(softmaxes, 1)
            accuracies = predictions.eq(labels)
            n_bins = 15
            
            n, bin_boundaries = np.histogram(confidences.cpu().detach(), histedges_equalN(confidences.cpu().detach(), n_bins=n_bins))
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            numerator = []
            dominator = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                if any(in_bin):
                    accuracies_temp = accuracies[in_bin]
                    accuracy_in_bin = accuracies_temp.float().mean().item()
                    confidence_in_bin = confidences[in_bin].mean()
                    numerator.append((confidence_in_bin - 1 / n_classes) * (accuracy_in_bin - 1 / n_classes))
                    dominator.append((confidence_in_bin - 1 / n_classes) * (confidence_in_bin - 1 / n_classes))
            
            numerator = torch.Tensor(numerator)
            dominator = torch.Tensor(dominator)
            self.weight = torch.sum(numerator) / torch.sum(dominator)
        
        else:
            T = 0.001
            for i in range(1000):
                temp_softmaxes = softmaxes.clone()
                self.weight = T
                temp_softmaxes = T * temp_softmaxes + (1 - T) * 1 / n_classes
                temp_softmaxes /= torch.sum(temp_softmaxes, 1, keepdim=True)
                after_temperature_ece = ece_criterion(temp_softmaxes, labels, is_logits=False).item()

                if ece_val > after_temperature_ece + eps:
                    T_opt_ece = T
                    ece_val = after_temperature_ece
                T += 0.001
            
            self.weight = T_opt_ece
        
        softmaxes = self.weight * softmaxes + (1 - self.weight) * 1 / n_classes
        softmaxes /= torch.sum(softmaxes, 1, keepdim=True)

        # Calculate ECE after temperature scaling
        after_temperature_ece = ece_criterion(softmaxes, labels, is_logits=False).item()
        if log:
            print('Optimal weight: %.3f' % self.weight)
            print('After temperature - ECE: %.3f' % after_temperature_ece)

        
        return self
    
            
def temperature_scale2(logits, temperature):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits
    return logits / temperature

def class_temperature_scale2(logits, csece_temperature):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits
    return logits / csece_temperature
       
def set_temperature2(logits, labels, iters=1, cross_validate='ece',
                     init_temp=2.5, acc_check=False, const_temp=False, log=True, num_bins=25):
    """
    Tune the tempearature of the model (using the validation set) with cross-validation on ECE or NLL
    """
    if const_temp:
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if log:
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.1
        for i in range(100):
            temperature = T
            after_temperature_nll = nll_criterion(temperature_scale2(logits, temperature), labels).item()
            after_temperature_ece = ece_criterion(temperature_scale2(logits, temperature), labels).item()
            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll

            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        if cross_validate == 'ece':
            temperature = T_opt_ece
        else:
            temperature = T_opt_nll

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(temperature_scale2(logits, temperature), labels).item()
        after_temperature_ece = ece_criterion(temperature_scale2(logits, temperature), labels).item()
        if log:
            print('Optimal temperature: %.3f' % temperature)
            print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    else:
        """
        Tune single tempearature for the model (using the validation set) with cross-validation on ECE
        """
        # Calculate ECE before temperature scaling
        ece_criterion = ECELoss(n_bins=num_bins).cuda()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if log:
            print('Before temperature - ECE: %.3f' % (before_temperature_ece))

        ece_val = 10 ** 7
        T_opt_ece = 1.0
        T = 0.1
        for i in range(100):
            temperature = T
            after_temperature_ece = ece_criterion(temperature_scale2(logits, temperature), labels).item()
            
            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        init_temp = T_opt_ece

        # Calculate NLL and ECE after temperature scaling
        after_temperature_ece = ece_criterion(temperature_scale2(logits, init_temp), labels).item()
        if log:
            print('Optimal temperature: %.3f' % init_temp)
            print('After temperature - ECE: %.3f' % (after_temperature_ece))
        
        """
        Find tempearature vector for the model (using the validation set) with cross-validation on ECE
        """
        #ece_criterion = estECELoss(n_bins=num_bins).cuda()
        ece_list = []
        
        # Calculate NLL and ECE before temperature scaling
        before_temperature_ece = ece_criterion(logits, labels).item()
        if acc_check:
            _, accuracy, _, _, _ = test_classification_net_logits(logits, labels)

        if log:
            print('Before temperature - ECE: {0:.3f}'.format(before_temperature_ece))

        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T_opt_csece = init_temp*torch.ones(logits.size()[1]).cuda()
        T_csece = init_temp*torch.ones(logits.size()[1]).cuda()
        csece_temperature = T_csece
        ece_list.append(ece_criterion(class_temperature_scale2(logits, csece_temperature), labels).item())
        if acc_check:
            _, temp_accuracy, _, _, _ = test_classification_net_logits(class_temperature_scale2(logits, csece_temperature), labels)
            if temp_accuracy >= accuracy:
                accuracy = temp_accuracy

        steps_limit = 0.2
        temp_steps = torch.linspace(-steps_limit, steps_limit, int((2 * steps_limit) / 0.1 + 1))
        ece_val = 10 ** 7
        csece_val = 10 ** 7
        converged = False
        prev_temperatures = csece_temperature.clone()
        for iter in range(iters):
            print('Started iter ' + str(iter))
        #while not converged:
            for label in range(logits.size()[1]):
                #init_temp_value = T_csece[label].item()
                T = 0.1
                for i in range(100):
                #for step in temp_steps:
                    T_csece[label] = T
                    #T_csece[label] = init_temp_value + step
                    csece_temperature = T_csece
                    temperature = T
                    after_temperature_ece = ece_criterion(class_temperature_scale2(logits, csece_temperature), labels).item()
                    
                    if acc_check:
                        _, temp_accuracy, _, _, _ = test_classification_net_logits(class_temperature_scale2(logits, csece_temperature), labels)

                    if acc_check:
                        if csece_val > after_temperature_ece and temp_accuracy >= accuracy:
                            T_opt_csece[label] = T
                            csece_val = after_temperature_ece
                            accuracy = temp_accuracy
                    else:
                        if csece_val > after_temperature_ece:
                            #T_opt_csece[label] = init_temp_value + step
                            T_opt_csece[label] = T
                            csece_val = after_temperature_ece
                    T += 0.1
                T_csece[label] = T_opt_csece[label]
            csece_temperature = T_opt_csece
            ece_list.append(ece_criterion(class_temperature_scale2(logits, csece_temperature), labels).item())
            #converged = torch.all(csece_temperature.eq(prev_temperatures))
            #prev_temperatures = csece_temperature.clone()
    csece_temperature = T_opt_csece
    
    if const_temp:
        return temperature
    else:
        return csece_temperature, init_temp


def bins_temperature_scale2(logits, bece_temperature):
        """
        Perform temperature scaling on logits
        """     
        # Expand temperature to match the size of logits
        return logits / torch.unsqueeze(bece_temperature, -1)


def histedges_equalN(x, n_bins=15):
    npt = len(x)
    return np.interp(np.linspace(0, npt, n_bins + 1),
                    np.arange(npt),
                    np.sort(x))


def equal_bins(x, n_bins=15):
    #sorted_samples = np.unique(x.numpy())
    #n, bin_boundaries = np.histogram(x, histedges_equalN(x, n_bins=n_bins))
    bin_size = int(x.shape[0] / n_bins)
    sorted_samples = np.sort(x)
    unique_samples, counts = np.unique(sorted_samples, return_counts=True)
    many_samples = dict()
    for sample, count in zip(unique_samples, counts):
        if count > bin_size:
            many_samples[sample] = count

    bin_boundaries = np.zeros(n_bins + 1)
    bin_boundaries[0] = sorted_samples[0]
    #bin_boundaries[-1] = 0.9999999
    bin_boundaries[-1] = 1.0
    counter = 0
    i = 1
    for sample in sorted_samples:
        if counter == bin_size*i:
            bin_boundaries[i] = sample
            i+=1
        counter+=1

    return bin_boundaries, many_samples


def bin_ece(logits, accuracies, in_bin, is_logits=True):
    accuracies_temp = accuracies[in_bin]
    origin_accuracy_in_bin = accuracies_temp.float().mean().item()
    accuracy_in_bin = min(origin_accuracy_in_bin, 0.99)
    accuracy_in_bin = max(accuracy_in_bin, 0.01)
    prop_in_bin = in_bin.float().mean()
    if is_logits:
        softmaxes_temp = F.softmax(logits, dim=1)
        confidences_temp, _ = torch.max(softmaxes_temp, 1)
    else:
        confidences_temp, _ = torch.max(logits, 1)
    avg_confidence_in_bin = confidences_temp.mean()
    after_temperature = torch.abs(accuracy_in_bin - avg_confidence_in_bin)
    samples = logits.shape[0]
    ece = (prop_in_bin * after_temperature).item()

    return ece, samples, origin_accuracy_in_bin, avg_confidence_in_bin


def bins_temperature_scale_test3(logits, labels, bins_T, iters, bin_boundaries, many_samples, single_temp, best_iter, n_bins=15):
        """
        Perform temperature scaling on logits
        """
        ece_criterion = ECELoss(n_bins=25).cuda()
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        #confidences[confidences > 0.9995] = 0.9995
        logits_np = logits.cpu().detach().numpy()
        scaled_logits = logits.clone()
        ece_list = []
        ece_per_bin = []
        single_ece_per_bin = []
        original_ece_per_bin = []
        print(f'Number of iters: {best_iter + 1}')
        for i in range(best_iter + 1):
            bin = 0
            prev_bin = None
            print('\n')            
            bin_lowers = bin_boundaries[i][:-1]
            bin_uppers = bin_boundaries[i][1:]
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                if any(in_bin):
                    """
                    # Smoothing
                    bin_len = max(bin_upper - bin_lower, 1e-5)
                    conf_location = (confidences[in_bin] - bin_lower) / bin_len
                    high_conf = conf_location.gt(0.5)
                    low_conf = conf_location.le(0.5)
                    if bin > 0 and bin < n_bins - 1:
                        prev_avg = (bins_T[bin, i] + bins_T[bin - 1, i]) / 2
                        next_avg = (bins_T[bin + 1, i] + bins_T[bin, i]) / 2
                        temps_high = (2 * (conf_location[high_conf] - 0.5) * abs(bins_T[bin, i] - next_avg) + bins_T[bin, i]).unsqueeze(dim=-1)
                        temps_low = (2 * conf_location[low_conf] * abs(bins_T[bin, i] - prev_avg) + prev_avg).unsqueeze(dim=-1)
                    elif bin == 0:
                        next_avg = (bins_T[bin + 1, i] + bins_T[bin, i]) / 2
                        temps_high = (2 * (conf_location[high_conf] - 0.5) * abs(bins_T[bin, i] - next_avg) + bins_T[bin, i]).unsqueeze(dim=-1)
                        temps_low = bins_T[bin, i]
                    else:
                        prev_avg = abs(bins_T[bin, i] + bins_T[bin - 1, i]) / 2
                        temps_high = bins_T[bin, i]
                        temps_low = (2 * conf_location[low_conf] * abs(bins_T[bin, i] - prev_avg) + prev_avg).unsqueeze(dim=-1)
                    #temps[temps == 0] = 1e-5
                    #scaled_logits[in_bin][high_conf] = scaled_logits[in_bin][high_conf] / temps_high
                    #scaled_logits[in_bin][low_conf] = scaled_logits[in_bin][low_conf] / temps_low
                    """
                    original_ece, samples, accuracy_in_bin, origin_avg_confidence_in_bin = bin_ece(logits[in_bin], accuracies, in_bin)
                    original_ece_per_bin.append(original_ece)
                    scaled_logits[in_bin] = scaled_logits[in_bin] / bins_T[bin, i]
                    ece, _, _, _ = bin_ece(scaled_logits[in_bin], accuracies, in_bin)
                    ece_per_bin.append(ece)
                    single_logits = logits[in_bin] / single_temp
                    single_ece, _, _, _ = bin_ece(single_logits, accuracies, in_bin)
                    single_ece_per_bin.append(single_ece)
                    print('original average confidence in bin ', bin + 1, ' :', origin_avg_confidence_in_bin.item())
                    print('ece in bin ', bin + 1, ' :', ece,
                        ', number of samples: ', samples)
                    print('accuracy in bin ', bin + 1, ': ', accuracy_in_bin)
                bin += 1

            ece_list.append(ece_criterion(scaled_logits, labels).item())
            softmaxes = F.softmax(scaled_logits, dim=1)
            confidences, _ = torch.max(softmaxes, 1)
        
        print(ece_list)
                            
        return scaled_logits, ece_per_bin, single_ece_per_bin, original_ece_per_bin, ece_list

def set_temperature3(logits, labels, iters=1, cross_validate='ece',
                     init_temp=2.5, const_temp=False, log=True, num_bins=25):
    """
    Tune the tempearature of the model (using the validation set) with cross-validation on ECE or NLL
    """
    if const_temp:
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if log:
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.1
        for i in range(100):
            temperature = T
            after_temperature_nll = nll_criterion(temperature_scale2(logits, temperature), labels).item()
            after_temperature_ece = ece_criterion(temperature_scale2(logits, temperature), labels).item()
            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll

            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        if cross_validate == 'ece':
            temperature = T_opt_ece
        else:
            temperature = T_opt_nll

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(temperature_scale2(logits, temperature), labels).item()
        after_temperature_ece = ece_criterion(temperature_scale2(logits, temperature), labels).item()
        if log:
            print('Optimal temperature: %.3f' % temperature)
            print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    else:
        """
        Tune single tempearature for the model (using the validation set) with cross-validation on ECE
        """
        # Calculate ECE before temperature scaling
        ece_criterion = ECELoss(n_bins=num_bins).cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if log:
            print('Before temperature - ECE: %.3f' % (before_temperature_ece))
            
        n_bins = num_bins
        if cross_validate != 'ece':
            n_bins = 50
        eps = 1e-5
        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.1
        labels = labels.type(torch.LongTensor).cuda()
        temps_iters = torch.ones(iters).cuda()
        for i in range(iters):
            temp_logits = logits.clone()
            for t in range(100):
                temperature = T
                after_temperature_ece = ece_criterion(temperature_scale2(temp_logits, temperature), labels).item()
                after_temperature_nll = nll_criterion(temperature_scale2(temp_logits, temperature), labels).item()
                if ece_val > after_temperature_ece:
                    T_opt_ece = T
                    ece_val = after_temperature_ece
                if nll_val > after_temperature_nll:
                    T_opt_nll = T
                    nll_val = after_temperature_nll
                T += 0.1
            if cross_validate == 'ece':
                temps_iters[i] = T_opt_ece
            else:
                temps_iters[i] = T_opt_nll
            temp_logits = temp_logits / T_opt_ece
            after_temperature_ece = ece_criterion(temperature_scale2(temp_logits, T_opt_ece), labels).item()
            print('Temperature for #{} iteration for single TS: {}'.format(i + 1, after_temperature_ece))
            
        if cross_validate == 'ece':
            temperature = T_opt_ece
        else:
            temperature = T_opt_nll

        init_temp = temperature
        
        # Calculate ECE after temperature scaling
        after_temperature_ece = ece_criterion(temperature_scale2(logits, temperature), labels).item()
        if log:
            print('Optimal temperature: %.3f' % init_temp)
            print('After temperature - ECE: %.3f' % (after_temperature_ece))

        init_temp = 1
        #top_temp = T_opt_ece
        
        bins_T = init_temp*torch.ones((n_bins, iters)).cuda()
        ece_list = []        
        ece_list.append(ece_criterion(temperature_scale2(logits, temperature), labels).item())
                
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        #confidences[confidences > 0.9995] = 0.9995
        accuracies = predictions.eq(labels)
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1).unsqueeze(0).repeat((iters, 1)).numpy()
        
        #steps_limit = 0.2
        #temp_steps = torch.linspace(-steps_limit, steps_limit, int((2 * steps_limit) / 0.1 + 1)).cuda()
        many_samples = None
        original_bins = torch.zeros(confidences.shape)
        ece_ada_list = []
        count_high_acc = 0
        is_acc = False
        n, bin_boundaries[0] = np.histogram(confidences.cpu().detach(), histedges_equalN(confidences.cpu().detach(), n_bins=n_bins))
        bin_lowers = bin_boundaries[0][:-1]
        bin_uppers = bin_boundaries[0][1:]
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            if any(in_bin):
                accuracies_temp = accuracies[in_bin]
                origin_accuracy_in_bin = accuracies_temp.float().mean().item()
                if origin_accuracy_in_bin > 0.99:
                    count_high_acc += 1
        if count_high_acc > int(n_bins/2):  # model is highly accurated
            is_acc = True
            confidences[confidences > 0.9995] = 0.9995

        for i in range(iters):
            if cross_validate == 'ece':
                T_opt_bece = init_temp*torch.ones(logits.shape[0]).cuda()
                T_bece = init_temp*torch.ones(logits.shape[0]).cuda()
                bece_temperature = T_bece
            else:
                T_opt_nll = init_temp*torch.ones(logits.shape[0]).cuda()
                T_nll = init_temp*torch.ones(logits.shape[0]).cuda()
                nll_temperature = T_nll
            
            ece_in_iter = 0
            print('iter num ', i+1)
            bin = 0
            few_examples = dict()
            if i == 0:
                n, bin_boundaries[i] = np.histogram(confidences.cpu().detach(), histedges_equalN(confidences.cpu().detach(), n_bins=n_bins))
            else:
                bin_boundaries[i] = bin_boundaries[i - 1]
            
            if cross_validate != 'ece':
                bin_boundaries[i][bin_boundaries[i] > 0.999] = 1
            
            #bin_boundaries = torch.linspace(0, 1, n_bins + 1)
            #bin_boundaries[i], many_samples = equal_bins(confidences.cpu().detach(), n_bins=n_bins)
            bin_lowers = bin_boundaries[i][:-1]
            bin_uppers = bin_boundaries[i][1:]

            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if confidences[in_bin].shape[0] < 20 and cross_validate == 'ece':
                    samples = T_bece[in_bin].shape[0]
                    print('number of samples in bin {0}: {1}'.format(bin + 1, samples))
                    few_examples[bin] = samples
                    bin += 1
                    continue
                if any(in_bin):
                    #init_temp_value = T_bece[in_bin][0].item()
                    T = 0.1
                    accuracies_temp = accuracies[in_bin]
                    origin_accuracy_in_bin = accuracies_temp.float().mean().item()
                    origin_avg_confidence_in_bin = confidences[in_bin].mean()
                    accuracy_in_bin = min(origin_accuracy_in_bin, 0.99)
                    accuracy_in_bin = max(accuracy_in_bin, 0.01)
                    
                    if is_acc and cross_validate == 'ece':
                        accuracy_in_bin = origin_accuracy_in_bin
                        
                    if cross_validate == 'ece':
                        bece_val = torch.abs(accuracy_in_bin - origin_avg_confidence_in_bin)
                    else:
                        nll_val = nll_criterion(logits[in_bin] / bins_T[bin, i], labels[in_bin]).item()
                    for t in range(100):
                    #for step in temp_steps:
                        #T_bece[in_bin] = init_temp_value + step
                        
                        logits_temp = logits.clone()
                        if cross_validate == 'ece':
                            T_bece[in_bin] = T
                            bece_temperature = T_bece
                            logits_temp[in_bin] = logits[in_bin] / torch.unsqueeze(T_bece[in_bin], -1)
                            softmaxes_temp = F.softmax(logits_temp, dim=1)
                            confidences_temp, _ = torch.max(softmaxes_temp[in_bin], 1)
                            avg_confidence_in_bin = confidences_temp.mean()
                            after_temperature = torch.abs(accuracy_in_bin - avg_confidence_in_bin)
                            
                            if bece_val > after_temperature + eps:
                                #T_opt_bece[in_bin] = init_temp_value + step
                                T_opt_bece[in_bin] = T
                                bece_val = after_temperature                      
                        
                        else:
                            T_nll[in_bin] = T
                            nll_temperature = T_nll
                            after_temperature_nll = nll_criterion(logits[in_bin] / torch.unsqueeze(T_nll[in_bin], -1), labels[in_bin]).item() 
                            
                            if nll_val > after_temperature_nll:
                                T_opt_nll[in_bin] = T
                                nll_val = after_temperature_nll
                        
                        T += 0.1
                      
                    original_bins[in_bin] = bin
                    if cross_validate == 'ece':
                        T_bece[in_bin] = T_opt_bece[in_bin]
                        bins_T[bin, i] = T_opt_bece[in_bin][0].item()
                        samples = T_bece[in_bin].shape[0]
                        ece_in_iter += prop_in_bin * bece_val
                    else:
                        T_nll[in_bin] = T_opt_nll[in_bin]
                        bins_T[bin, i] = T_opt_nll[in_bin][0].item()
                        samples = T_nll[in_bin].shape[0]
                        ece_in_iter += prop_in_bin * nll_val
                                            
                    print('original average confidence in bin ', bin + 1, ' :', origin_avg_confidence_in_bin.item())
                    if cross_validate == 'ece':
                        print('ece in bin ', bin+1, ' :', (prop_in_bin * bece_val).item(), ', number of samples: ', samples)
                    else:
                        print('ece in bin ', bin+1, ' :', (prop_in_bin * nll_val).item(), ', number of samples: ', samples)
                    print('accuracy in bin ', bin+1, ': ', origin_accuracy_in_bin)

                bin += 1

            print(bins_T[:, i])
            if cross_validate == 'ece':
                for bin in few_examples:
                    #bins_T[bin, i] = temperature

                    if bin > 0 and bin < n_bins - 1:
                        lower_bin = bin - 1
                        upper_bin = bin + 1
                        while lower_bin in few_examples and lower_bin - 1 >= 0:
                            lower_bin -= 1
                        while upper_bin in few_examples and upper_bin + 1 <= n_bins - 1:
                            upper_bin += 1
                        if upper_bin == n_bins - 1:
                            bins_T[bin, i] = bins_T[lower_bin, i]
                        else:
                            avg_temp = (bins_T[lower_bin, i] + bins_T[upper_bin, i]) / 2  # Mean temperature of neighbors
                            bins_T[bin, i] = avg_temp
                    elif bin == 0:
                        upper_bin = bin + 1
                        while upper_bin in few_examples and upper_bin + 1 <= n_bins - 1:
                            upper_bin += 1
                        bins_T[bin, i] = bins_T[upper_bin, i]
                    else:
                        lower_bin = bin - 1
                        while lower_bin in few_examples and lower_bin - 1 >= 0:
                            lower_bin -= 1
                        bins_T[bin, i] = bins_T[lower_bin, i]
            
            if cross_validate == 'ece':
                bece_temperature = T_opt_bece
                current_ece = ece_criterion(bins_temperature_scale2(logits, bece_temperature), labels).item()
            else:
                nll_temperature = T_opt_nll
                current_ece = ece_criterion(bins_temperature_scale2(logits, nll_temperature), labels).item()
            print('ece in iter ', i+1, ' :', current_ece)
            if i > 0 and current_ece < ece_list[best_iter]:
                best_iter = i
            if i == 0:
                best_iter = 0
            if abs(ece_list[-1] - current_ece) > eps:
                ece_list.append(current_ece)
            else:
                iters = i + 1
                break

            ece_ada_list.append(ece_in_iter.item())
            if cross_validate == 'ece':
                logits = logits / torch.unsqueeze(bece_temperature, -1)
            else:
                logits = logits / torch.unsqueeze(nll_temperature, -1)
            softmaxes = F.softmax(logits, dim=1)
            confidences, _ = torch.max(softmaxes, 1)
            moved_bins = torch.zeros(confidences.shape)
            bin = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                moved_bins[in_bin] = bin
                bin += 1
            bins_moved = torch.eq(original_bins, moved_bins)
            moved_precentage = bins_moved.float().mean()
            print('Precentage of moved bins after scaling: ', 100 - (moved_precentage * 100).item())
        
        if const_temp:
            return temperature
        else:
            return bins_T, temperature, bin_boundaries, many_samples, best_iter

def check_movements(logits, const):
    softmaxes = F.softmax(logits, dim=1)
    original_confidences, _ = torch.max(softmaxes, 1)
    before_indices = torch.argsort(original_confidences)
    moved_softmaxes = F.softmax(logits / const, dim=1)
    moved_confidences, _ = torch.max(moved_softmaxes, 1)
    after_indices = torch.argsort(moved_confidences)
    
    return before_indices, after_indices

def bins_temperature_scale_test4(logits, labels, bins_T, iters, bin_boundaries, single_temp, best_iter, n_bins=15):
        """
        Perform temperature scaling on logits
        """
        ece_criterion = ECELoss(n_bins=25).cuda()
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        # confidences[confidences > 0.99999] = 0.99999
        logits_np = logits.cpu().detach().numpy()
        n_classes = logits.shape[1]
        ece_list = []
        ece_per_bin = []
        single_ece_per_bin = []
        original_ece_per_bin = []
        print(f'Number of iters: {best_iter + 1}')
        for i in range(best_iter + 1):
            bin = 0
            prev_bin = None
            print('\n')            
            bin_lowers = bin_boundaries[i][:-1]
            bin_uppers = bin_boundaries[i][1:]
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                if any(in_bin):
                    original_ece, samples, accuracy_in_bin, origin_avg_confidence_in_bin = bin_ece(logits[in_bin], accuracies, in_bin)
                    original_ece_per_bin.append(original_ece)
                    weight = bins_T[bin, i]
                    softmaxes[in_bin] = weight * softmaxes[in_bin] + (1 - weight) * 1 / n_classes
                    ece, _, _, _ = bin_ece(softmaxes[in_bin], accuracies, in_bin, is_logits=False)
                    ece_per_bin.append(ece)
                    single_logits = logits[in_bin] / single_temp
                    single_ece, _, _, _ = bin_ece(single_logits, accuracies, in_bin)
                    single_ece_per_bin.append(single_ece)
                    print('original average confidence in bin ', bin + 1, ' :', origin_avg_confidence_in_bin.item())
                    print('ece in bin ', bin + 1, ' :', ece,
                        ', number of samples: ', samples)
                    print('accuracy in bin ', bin + 1, ': ', accuracy_in_bin)
                bin += 1

            softmaxes /= torch.sum(softmaxes, 1, keepdim=True)
            ece_list.append(ece_criterion(softmaxes, labels, is_logits=False).item())
            confidences, _ = torch.max(softmaxes, 1)
        
        print(ece_list)
                            
        return softmaxes, ece_per_bin, single_ece_per_bin, original_ece_per_bin, ece_list

def set_temperature4(logits, labels, iters=1, cross_validate='ece',
                     init_temp=2.5, acc_check=False, const_temp=False, log=True, num_bins=25, top_temp=10):
    """
    Tune the tempearature of the model (using the validation set) with cross-validation on ECE or NLL
    """
    if const_temp:
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if log:
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.1
        for i in range(100):
            temperature = T
            after_temperature_nll = nll_criterion(temperature_scale2(logits, temperature), labels).item()
            after_temperature_ece = ece_criterion(temperature_scale2(logits, temperature), labels).item()
            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll

            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        if cross_validate == 'ece':
            temperature = T_opt_ece
        else:
            temperature = T_opt_nll

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(temperature_scale2(logits, temperature), labels).item()
        after_temperature_ece = ece_criterion(temperature_scale2(logits, temperature), labels).item()
        if log:
            print('Optimal temperature: %.3f' % temperature)
            print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    else:
        """
        Tune single tempearature for the model (using the validation set) with cross-validation on ECE
        """
        # Calculate ECE before temperature scaling
        ece_criterion = ECELoss(n_bins=num_bins).cuda()
        nll_criterion = nn.NLLLoss().cuda()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if log:
            print('Before temperature - ECE: %.3f' % (before_temperature_ece))
            
        n_bins = num_bins
        eps = 1e-5
        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.1
        labels = labels.type(torch.LongTensor).cuda()
        temps_iters = torch.ones(iters).cuda()
        for i in range(iters):
            temp_logits = logits.clone()
            for t in range(100):
                temperature = T
                after_temperature_ece = ece_criterion(temperature_scale2(temp_logits, temperature), labels).item()
                after_temperature_nll = nll_criterion(temperature_scale2(temp_logits, temperature), labels).item()
                if ece_val > after_temperature_ece:
                    T_opt_ece = T
                    ece_val = after_temperature_ece
                if nll_val > after_temperature_nll:
                    T_opt_nll = T
                    nll_val = after_temperature_nll
                T += 0.1
            temps_iters[i] = T_opt_ece
            temp_logits = temp_logits / T_opt_ece
            after_temperature_ece = ece_criterion(temperature_scale2(temp_logits, T_opt_ece), labels).item()
            print('Temperature for #{} iteration for single TS: {}'.format(i + 1, after_temperature_ece))
            
        temperature = T_opt_ece
        init_temp = temperature
        
        # Calculate ECE after temperature scaling
        after_temperature_ece = ece_criterion(temperature_scale2(logits, temperature), labels).item()
        if log:
            print('Optimal temperature: %.3f' % init_temp)
            print('After temperature - ECE: %.3f' % (after_temperature_ece))

        init_temp = 0
        
        bins_T = init_temp*torch.ones((n_bins, iters)).cuda()
        ece_list = []        
        ece_list.append(ece_criterion(temperature_scale2(logits, temperature), labels).item())
                
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        # confidences[confidences > 0.99999] = 0.99999
        accuracies = predictions.eq(labels)
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1).unsqueeze(0).repeat((iters, 1)).numpy()
        
        original_bins = torch.zeros(confidences.shape)
        ece_ada_list = []
        count_high_acc = 0
        is_acc = False
        n, bin_boundaries[0] = np.histogram(confidences.cpu().detach(), histedges_equalN(confidences.cpu().detach(), n_bins=n_bins))
        bin_lowers = bin_boundaries[0][:-1]
        bin_uppers = bin_boundaries[0][1:]
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            if any(in_bin):
                accuracies_temp = accuracies[in_bin]
                origin_accuracy_in_bin = accuracies_temp.float().mean().item()
                if origin_accuracy_in_bin > 0.99:
                    count_high_acc += 1
        if count_high_acc > int(n_bins/2):  # model is highly accurated
            is_acc = True
            confidences[confidences > 0.9999] = 0.9999

        n_classes = logits.shape[1]
        
        conf_acc_diff = []
        
        for i in range(iters):
            T_opt_bece = init_temp*torch.ones(logits.shape[0]).cuda()
            T_bece = init_temp*torch.ones(logits.shape[0]).cuda()
            bece_temperature = T_bece
            
            ece_in_iter = 0
            print('iter num ', i+1)
            bin = 0
            few_examples = dict()
            if i == 0:
                n, bin_boundaries[i] = np.histogram(confidences.cpu().detach(), histedges_equalN(confidences.cpu().detach(), n_bins=n_bins))
            else:
                bin_boundaries[i] = bin_boundaries[i - 1]
            
            bin_lowers = bin_boundaries[i][:-1]
            bin_uppers = bin_boundaries[i][1:]
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if confidences[in_bin].shape[0] < 20:
                    samples = T_bece[in_bin].shape[0]
                    print('number of samples in bin {0}: {1}'.format(bin + 1, samples))
                    few_examples[bin] = samples
                    if bin == 0:
                        conf_acc_diff.append(0)
                    else:
                        conf_acc_diff.append(conf_acc_diff[-1])
                    bin += 1
                    continue
                if any(in_bin):
                    T = 0.0
                    accuracies_temp = accuracies[in_bin]
                    origin_accuracy_in_bin = accuracies_temp.float().mean().item()
                    origin_avg_confidence_in_bin = confidences[in_bin].mean()
                    accuracy_in_bin = min(origin_accuracy_in_bin, 0.99)
                    accuracy_in_bin = max(accuracy_in_bin, 0.01)
                    
                    conf_acc_diff.append(origin_avg_confidence_in_bin - origin_accuracy_in_bin)
                    
                    if is_acc:
                        accuracy_in_bin = origin_accuracy_in_bin
                        
                    """
                    bece_val = torch.abs(accuracy_in_bin - origin_avg_confidence_in_bin)
                    # bece_val = ece_criterion(logits, labels)
                    for t in range(100):
                        temp_softmaxes = softmaxes.clone()
                        T_bece[in_bin] = T
                        bece_temperature = T_bece
                        temp_softmaxes[in_bin] = T * temp_softmaxes[in_bin] + (1 - T) * 1 / n_classes
                        # temp_softmaxes[in_bin] = temp_softmaxes[in_bin] - T * dists[in_bin]
                        temp_softmaxes /= torch.sum(temp_softmaxes, 1, keepdim=True)
                        confidences_temp, _ = torch.max(temp_softmaxes[in_bin], 1)
                        avg_confidence_in_bin = confidences_temp.mean()
                        after_temperature = torch.abs(accuracy_in_bin - avg_confidence_in_bin)
                        # after_temperature = ece_criterion(temp_softmaxes, labels, is_logits=False)
                        
                        if bece_val > after_temperature + eps:
                            T_opt_bece[in_bin] = T
                            bece_val = after_temperature                      
                        
                        T += 0.01
                    """
                    
                    weight_i = (origin_accuracy_in_bin - 1/n_classes) / (origin_avg_confidence_in_bin - 1/n_classes)
                    # weight_i = max(0, min(1, weight_i.item()))
                    T_opt_bece[in_bin] = weight_i
                    
                    # Compute bin's updated ECE
                    temp_softmaxes = softmaxes.clone()
                    temp_softmaxes[in_bin] = weight_i * temp_softmaxes[in_bin] + (1 - weight_i) * 1 / n_classes
                    temp_softmaxes /= torch.sum(temp_softmaxes, 1, keepdim=True)
                    confidences_temp, _ = torch.max(temp_softmaxes[in_bin], 1)
                    avg_confidence_in_bin = confidences_temp.mean()
                    bece_val = torch.abs(origin_accuracy_in_bin - avg_confidence_in_bin)
                    
                    original_bins[in_bin] = bin
                    T_bece[in_bin] = T_opt_bece[in_bin]
                    bins_T[bin, i] = T_opt_bece[in_bin][0].item()
                    samples = T_bece[in_bin].shape[0]
                    ece_in_iter += prop_in_bin * bece_val
                                            
                    print('original average confidence in bin ', bin + 1, ' :', origin_avg_confidence_in_bin.item())
                    print('ece in bin ', bin+1, ' :', (prop_in_bin * bece_val).item(), ', number of samples: ', samples)
                    print('accuracy in bin ', bin+1, ': ', origin_accuracy_in_bin)
                      

                bin += 1

            print(bins_T[:, i])
            for bin in few_examples:
                #bins_T[bin, i] = temperature

                if bin > 0 and bin < n_bins - 1:
                    lower_bin = bin - 1
                    upper_bin = bin + 1
                    while lower_bin in few_examples and lower_bin - 1 >= 0:
                        lower_bin -= 1
                    while upper_bin in few_examples and upper_bin + 1 <= n_bins - 1:
                        upper_bin += 1
                    if upper_bin == n_bins - 1:
                        bins_T[bin, i] = bins_T[lower_bin, i]
                    else:
                        avg_temp = (bins_T[lower_bin, i] + bins_T[upper_bin, i]) / 2  # Mean temperature of neighbors
                        bins_T[bin, i] = avg_temp
                elif bin == 0:
                    upper_bin = bin + 1
                    while upper_bin in few_examples and upper_bin + 1 <= n_bins - 1:
                        upper_bin += 1
                    bins_T[bin, i] = bins_T[upper_bin, i]
                else:
                    lower_bin = bin - 1
                    while lower_bin in few_examples and lower_bin - 1 >= 0:
                        lower_bin -= 1
                    bins_T[bin, i] = bins_T[lower_bin, i]
            
            bece_temperature = T_opt_bece

            ece_ada_list.append(ece_in_iter.item())
            weight = bece_temperature.unsqueeze(-1).repeat(1, logits.shape[1])
            softmaxes = weight * softmaxes + (1 - weight) * 1 / n_classes
            # softmaxes = softmaxes - bece_temperature.unsqueeze(-1).repeat(1, dists.shape[1]) * dists
            confidences, _ = torch.max(softmaxes, 1)
            current_ece = ece_criterion(softmaxes, labels, is_logits=False).item()
                
            print('ece in iter ', i+1, ' :', current_ece)
            if i > 0 and current_ece < ece_list[best_iter]:
                best_iter = i
            if i == 0:
                best_iter = 0
            if abs(ece_list[-1] - current_ece) > eps:
                ece_list.append(current_ece)
            else:
                iters = i + 1
                break
            
            moved_bins = torch.zeros(confidences.shape)
            bin = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                moved_bins[in_bin] = bin
                bin += 1
            bins_moved = torch.eq(original_bins, moved_bins)
            moved_precentage = bins_moved.float().mean()
            print('Precentage of moved bins after scaling: ', 100 - (moved_precentage * 100).item())
        
        if const_temp:
            return temperature
        else:
            return bins_T, temperature, bin_boundaries, best_iter, conf_acc_diff
        
def bins_temperature_scale_test5(logits, weight):
        """
        Perform weight scaling on logits
        """
        softmaxes = F.softmax(logits, 1)
        n_classes = logits.shape[1]
        
        softmaxes = weight * softmaxes + (1 - weight) * 1 / n_classes
                                    
        return softmaxes

def set_temperature5(logits, labels, log=True):
    """
    Tune the tempearature of the model (using the validation set) with cross-validation on ECE without bins
    """
    ece_criterion = ECELoss().cuda()

    # Calculate ECE before temperature scaling
    before_weight_ece = ece_criterion(logits, labels).item()
    if log:
        print('Before weight scaling - ECE: %.3f' % before_weight_ece)
    
    softmaxes = F.softmax(logits, 1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    n_classes = logits.shape[1]
    n_bins = 13
    
    ece_val = 10 ** 7
    T_opt_ece = 1.0
    n, bin_boundaries = np.histogram(confidences.cpu().detach(), histedges_equalN(confidences.cpu().detach(), n_bins=n_bins))
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    numerator = []
    dominator = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        if any(in_bin):
            accuracies_temp = accuracies[in_bin]
            accuracy_in_bin = accuracies_temp.float().mean().item()
            confidence_in_bin = confidences[in_bin].mean()
            numerator.append((confidence_in_bin - 1 / n_classes) * (accuracy_in_bin - 1 / n_classes))
            dominator.append((confidence_in_bin - 1 / n_classes) * (confidence_in_bin - 1 / n_classes))
            
    """
    T = 0.01
    for i in range(100):
        temp_softmaxes = softmaxes.clone()
        weight = T
        temp_softmaxes = temp_softmaxes - T * dists
        temp_softmaxes /= torch.sum(temp_softmaxes, 1, keepdim=True)
        after_weight_ece = ece_criterion(temp_softmaxes, labels, is_logits=False).item()

        if ece_val > after_weight_ece:
            T_opt_ece = T
            ece_val = after_weight_ece
        T += 0.01
    """
    
    numerator = torch.Tensor(numerator)
    dominator = torch.Tensor(dominator)
    weight = torch.sum(numerator) / torch.sum(dominator)
    
    softmaxes = weight * softmaxes + (1 - weight) * 1 / n_classes

    # Calculate ECE after temperature scaling
    after_weight_ece = ece_criterion(softmaxes, labels, is_logits=False).item()
    if log:
        print('Optimal weight: %.3f' % weight)
        print('After weight scaling - ECE: %.3f' % after_weight_ece)

    
    return weight
