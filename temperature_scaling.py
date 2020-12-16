'''
Code to perform temperature scaling. Adapted from https://github.com/gpleiss/temperature_scaling
'''
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

from Metrics.metrics import ECELoss, ClassECELoss


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, log=True, const_temp=False):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = 1.0
        self.log = log
        self.const_temp = const_temp


    def forward(self, input):
        logits = self.model(input)
        if self.const_temp:
            return self.temperature_scale(logits)
        else:
            return self.class_temperature_scale(logits)


    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        return logits / self.temperature
    
    def class_temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        return logits / self.csece_temperature


    def set_temperature(self,
                        valid_loader, iters=1,
                        cross_validate='ece'):
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
            before_temperature_csece, _ = csece_criterion(logits, labels)
            if self.log:
                print('Before temperature - NLL: {0:.3f}, ECE: {1:.3f}, classECE: {2}'.format(before_temperature_nll, before_temperature_ece, before_temperature_csece))

            nll_val = 10 ** 7
            ece_val = 10 ** 7
            csece_val = 10 ** 7
            T_opt_nll = 1.0
            T_opt_ece = 1.0
            T_opt_csece = torch.full((1, logits.size()[1]), 2.5).cuda()
            T_csece = torch.full((1, logits.size()[1]), 2.5).cuda()
            for iter in range(iters):
                for label in range(logits.size()[1]):
                    T = 0.1
                    for i in range(100):
                        T_csece[label] = T
                        self.csece_temperature = T_csece
                        self.temperature = T
                        self.cuda()
                        after_temperature_nll = nll_criterion(self.class_temperature_scale(logits), labels).item()
                        after_temperature_ece = ece_criterion(self.class_temperature_scale(logits), labels).item()
                        after_temperature_csece, _ = csece_criterion(self.class_temperature_scale(logits), labels)
                        if nll_val > after_temperature_nll:
                            T_opt_nll = T
                            nll_val = after_temperature_nll

                        if ece_val > after_temperature_ece:
                            T_opt_ece = T
                            ece_val = after_temperature_ece

                        if csece_val > after_temperature_ece:
                            T_opt_csece[label] = T
                            csece_val = after_temperature_ece
                        T += 0.1
                    T_csece[label] = T_opt_csece[label]

            if cross_validate == 'ece':
                self.temperature = T_opt_ece
            else:
                self.temperature = T_opt_nll
            self.csece_temperature = T_opt_csece
            self.cuda()

            # Calculate NLL and ECE after temperature scaling
            after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            after_temperature_csece, _ = csece_criterion(self.class_temperature_scale(logits), labels)
            if self.log:
                print('Optimal temperature: %.3f' % self.temperature)
                print('After temperature - NLL: {0:.3f}, ECE: {1:.3f}, classECE: {2}'.format(after_temperature_nll, after_temperature_ece, after_temperature_csece))

        return self


    def get_temperature(self):
        if self.const_temp:
            return self.temperature
        else:
            return self.temperature, self.csece_temperature
