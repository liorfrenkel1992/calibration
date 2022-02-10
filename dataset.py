import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pickle
import random
import csv
# import pandas as pd


class LogitsLabelsDataset(Dataset):
    """logits and labels dataset."""

    def __init__(self, args, root_dir, type='val'):
        """
        Args:
            root_dir (string): Directory with all the logits and labels.
        """
        with open(root_dir + '/logits/' + args.model_name + '/' + type + '_labels.pickle', 'rb') as handle:
            self.labels = pickle.load(handle)
        with open(root_dir + '/logits/' + args.model_name + '/' + type + '_logits.pickle', 'rb') as handle:
            self.logits = pickle.load(handle)
            
    def __len__(self):
        return len(self.logits)

    def __getitem__(self, idx):
        
        return self.logits[idx], self.labels[idx].long()
    
class Covid19(Dataset):
    """Covid19 dataset"""
    
    def __init__(self, args, root_dir='COVID-19', type='train'):
        """
        Args:
            root_dir (string): Directory with all images.
        """
        self.trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self.path = root_dir
        
        with open(root_dir + '/' + type + '_labels.p', 'rb') as handle:
            self.labels = pickle.load(handle)
        with open(root_dir + '/' + type + '_imgs.p', 'rb') as handle:
            self.imgs = pickle.load(handle)
        
        # labels_names = os.listdir(root_dir)
        
        # self.labels = []
        # self.imgs = []
                
        # for inx, name in enumerate(labels_names):
        #     imgs = os.listdir(root_dir + '/' + name)
        #     for img in imgs:
        #         self.imgs.append(root_dir + '/' + name + '/' + img)
        #         self.labels.append(inx)
                
        # batch = list(zip(self.imgs, self.labels))
        # random.shuffle(batch)
        # self.imgs, self.labels = zip(*batch)
                      
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.trans(img)
        
        # if img.shape[0] < 3:
        #     img = img.repeat(3, 1, 1)
        
        # mean, std = img.mean([1,2]), img.std([1,2])
        # norm = transforms.Compose([transforms.Normalize(mean, std)])
        
        # img = norm(img)
            
        return img, self.labels[idx]
    
    
class Chexpert(Dataset):
    """Chexpert dataset"""
    
    def __init__(self, args, root_dir='vanilla_medical_classifier_chexpert/DataSet', type='train'):
        """
        Args:
            root_dir (string): Directory with all images.
        """
        # csv_file = type + '.csv'
        # df = pd.read_csv(root_dir + '/' + csv_file)
        # imgs = []
        # labels = []
        # for index, row in df.iterrows():
        #     print(index)
        #     if 1.0 in row.value_counts():
        #         if row.value_counts()[1.0] == 1:
        #             label = df.columns[np.array(row.eq(1.0))][0]
        #             imgs.append('vanilla_medical_classifier_chexpert/DataSet/' + row['Path'])
        #             labels.append(label)
                    
        # with open('Salary_Data.csv') as file:
        #     content = file.readlines()
        # header = content[:1]
        # rows = content[1:]
        # file = open(root_dir + '/' + csv_file)
        # csvreader = csv.reader(file)
        # header = []
        # header = next(csvreader)
        # rows = []
        # for row in csvreader:
        #     rows.append(row)
        # file.close()
        self.trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self.path = root_dir
        
        with open(root_dir + '/CheXpert-v1.0-small/' + type + '_labels.p', 'rb') as handle:
            self.labels = pickle.load(handle)
        with open(root_dir + '/CheXpert-v1.0-small/' + type + '_imgs.p', 'rb') as handle:
            self.imgs = pickle.load(handle)
        
        self.labels_dict = {'No Finding':0, 'Enlarged Cardiomediastinum': 1, 'Cardiomegaly':2, 'Lung Lesion':3, 'Lung Opacity':4, 
                            'Edema':5, 'Consolidation':6, 'Pneumonia':7, 'Atelectasis':8, 'Pneumothorax':9, 'Pleural Effusion':10,
                            'Pleural Other':11, 'Fracture':12, 'Support Devices':13}
        if args.n_classes == 14:
            new_labels = [self.labels_dict[label] for label in self.labels]
            self.labels = new_labels
        else:
            new_labels = [self.labels_dict[label] - 1 for label in self.labels]
            self.labels = new_labels
            if args.n_classes == 13:
                new_labels = self.labels.copy()
                inx = 0
                for label in new_labels:
                    if label == -1 or label == 3 or label == 4 or label == 12:
                        del self.labels[inx]
                        del self.imgs[inx]
                    else:
                        inx += 1
            
            elif args.n_classes == 12:
                new_labels = self.labels.copy()
                inx = 0
                for label in new_labels:
                    if label == -1 or label == 3:
                        del self.labels[inx]
                        del self.imgs[inx]
                    else:
                        inx += 1
                new_labels = [self.labels[inx] - 1 if self.labels[inx] >= 4 else self.labels[inx] for inx in range(len(self.labels))]
                self.labels = new_labels
            
            elif args.n_classes == 11:
                new_labels = self.labels.copy()
                inx = 0
                for label in new_labels:
                    if label == -1 or label == 3 or label == 4:
                        del self.labels[inx]
                        del self.imgs[inx]
                    else:
                        inx += 1
                new_labels = [self.labels[inx] - 2 if self.labels[inx] >= 5 else self.labels[inx] for inx in range(len(self.labels))]
                self.labels = new_labels
            elif args.n_classes == 10:
                new_labels = self.labels.copy()
                inx = 0
                for label in new_labels:
                    if label == -1 or label == 3 or label == 4 or label == 12:
                        del self.labels[inx]
                        del self.imgs[inx]
                    else:
                        inx += 1
                new_labels = [self.labels[inx] - 2 if self.labels[inx] >= 5 else self.labels[inx] for inx in range(len(self.labels))]
                self.labels = new_labels
            elif args.n_classes == 7:
                new_labels = self.labels.copy()
                inx = 0
                for label in new_labels:
                    if label == -1 or label == 3 or label == 4 or label == 12:
                        del self.labels[inx]
                        del self.imgs[inx]
                    else:
                        inx += 1
                new_labels = [self.labels[inx] - 2 if self.labels[inx] >= 5 else self.labels[inx] for inx in range(len(self.labels))]
                
                self.labels = new_labels.copy()
                inx = 0
                for label in new_labels:
                    if label == 2 or label == 4 or label == 9:
                        del self.labels[inx]
                        del self.imgs[inx]
                    else:
                        inx += 1
                
                new_labels = [self.labels[inx] - 1 if self.labels[inx] == 3 else self.labels[inx] for inx in range(len(self.labels))]
        
                new_labels = [new_labels[inx] - 2 if new_labels[inx] >= 5 else new_labels[inx] for inx in range(len(new_labels))]
                self.labels = new_labels
            
            elif args.n_classes == 6:
                new_labels = self.labels.copy()
                inx = 0
                for label in new_labels:
                    if label == -1 or label == 3 or label == 4 or label == 12:
                        del self.labels[inx]
                        del self.imgs[inx]
                    else:
                        inx += 1
                new_labels = [self.labels[inx] - 2 if self.labels[inx] >= 5 else self.labels[inx] for inx in range(len(self.labels))]
                
                self.labels = new_labels.copy()
                inx = 0
                for label in new_labels:
                    if label == 2 or label == 4 or label == 9:
                        del self.labels[inx]
                        del self.imgs[inx]
                    else:
                        inx += 1
                
                new_labels = [self.labels[inx] - 1 if self.labels[inx] == 3 else self.labels[inx] for inx in range(len(self.labels))]
        
                new_labels = [new_labels[inx] - 2 if new_labels[inx] >= 5 else new_labels[inx] for inx in range(len(new_labels))]
                
                self.labels = new_labels.copy()
                inx = 0
                for label in new_labels:
                    if label == 2:
                        del self.labels[inx]
                        del self.imgs[inx]
                    else:
                        inx += 1
                
                new_labels = [self.labels[inx] - 1 if self.labels[inx] >= 3 else self.labels[inx] for inx in range(len(self.labels))]
                self.labels = new_labels
        
        # labels_names = os.listdir(root_dir)
        
        # self.labels = []
        # self.imgs = []
                
        # for inx, name in enumerate(labels_names):
        #     imgs = os.listdir(root_dir + '/' + name)
        #     for img in imgs:
        #         self.imgs.append(root_dir + '/' + name + '/' + img)
        #         self.labels.append(inx)
                
        # batch = list(zip(self.imgs, self.labels))
        # random.shuffle(batch)
        # self.imgs, self.labels = zip(*batch)
                      
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open('/home/dsi/davidsr/' + img_path).convert('RGB')
        img = self.trans(img)
        
        # if img.shape[0] < 3:
        #     img = img.repeat(3, 1, 1)
        
        # mean, std = img.mean([1,2]), img.std([1,2])
        # norm = transforms.Compose([transforms.Normalize(mean, std)])
        
        # img = norm(img)
            
        return img, self.labels[idx]
    

class CXR14(Dataset):
    """CXR14 dataset"""
    
    def __init__(self, args, root_dir='/mnt/dsi_vol1/users/frenkel2/data/calibration/focal_calibration-1/CXR14', type='train'):
        """
        Args:
            root_dir (string): Directory with all images.
        """
        # csv_file = 'Data_Entry_2017_v2020.csv'
        # df = pd.read_csv(root_dir + '/' + csv_file)
        # imgs = []
        # labels = []
        # for index, row in df.iterrows():
        #     print(index)
        #     imgs.append(root_dir + '/images/' + row['Image Index'])
        #     labels.append(row['Finding Labels'].split('|')[0])
                    
        self.trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self.path = root_dir
        
        with open(root_dir + '/' + type + '_labels.p', 'rb') as handle:
            self.labels = pickle.load(handle)
        with open(root_dir + '/' + type + '_imgs.p', 'rb') as handle:
            self.imgs = pickle.load(handle)
        
        unique_labels = np.unique(self.labels)
        self.labels_dict = {}
        for inx, unique_label in enumerate(unique_labels):
            self.labels_dict[unique_label] = inx
            
        new_labels = [self.labels_dict[label] for label in self.labels]
        self.labels = new_labels
                      
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.trans(img)
        
        # if img.shape[0] < 3:
        #     img = img.repeat(3, 1, 1)
        
        # mean, std = img.mean([1,2]), img.std([1,2])
        # norm = transforms.Compose([transforms.Normalize(mean, std)])
        
        # img = norm(img)
            
        return img, self.labels[idx]
    

class HAM10000(Dataset):
    """HAM10000 dataset"""
    
    def __init__(self, args, root_dir='/mnt/dsi_vol1/users/frenkel2/data/calibration/focal_calibration-1/dataverse/dataverse_files', type='train'):
        """
        Args:
            root_dir (string): Directory with all images.
        """
        # csv_file = 'HAM10000_metadata'
        # df = pd.read_csv(root_dir + '/' + csv_file)
        # imgs = []
        # labels = []
        # for index, row in df.iterrows():
        #     print(index)
        #     imgs.append(root_dir + '/images/' + row['image_id'])
        #     labels.append(row['dx'])
                    
        self.trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self.path = root_dir
        
        with open(root_dir + '/' + type + '_labels.p', 'rb') as handle:
            self.labels = pickle.load(handle)
        with open(root_dir + '/' + type + '_imgs.p', 'rb') as handle:
            self.imgs = pickle.load(handle)
            
        # with open(root_dir + '/labels.p', 'rb') as handle:
        #     self.labels = pickle.load(handle)
        # with open(root_dir + '/imgs.p', 'rb') as handle:
        #     self.imgs = pickle.load(handle)
            
        # batch = list(zip(self.imgs, self.labels))
        # random.shuffle(batch)
        # self.imgs, self.labels = zip(*batch)
        
        unique_labels = np.unique(self.labels)
        self.labels_dict = {}
        for inx, unique_label in enumerate(unique_labels):
            self.labels_dict[unique_label] = inx
            
        new_labels = [self.labels_dict[label] for label in self.labels]
        self.labels = new_labels
                      
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path + '.jpg').convert('RGB')
        img = self.trans(img)
        
        # if img.shape[0] < 3:
        #     img = img.repeat(3, 1, 1)
        
        # mean, std = img.mean([1,2]), img.std([1,2])
        # norm = transforms.Compose([transforms.Normalize(mean, std)])
        
        # img = norm(img)
            
        return img, self.labels[idx]