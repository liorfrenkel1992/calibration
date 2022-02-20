import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pickle
import argparse


def parseArgs():
    load_split_path = '/mnt/dsi_vol1/users/frenkel2/data/calibration/focal_calibration-1/vanilla_medical_classifier_chexpert/DataSet'
    load_data_path = '/home/dsi/davidsr'

    parser = argparse.ArgumentParser(
        description="Train/test medical imaging datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--load_split_path", type=str, default=load_split_path, dest="load_split_path",
                        help="Path of splitted data")
    parser.add_argument("--load_data_path", type=str, default=load_data_path, dest="load_data_path",
                        help="Path of raw data")
    parser.add_argument("--ds_name", type=str, default="chexpert", dest="ds_name",
                        help="ds name")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "test"],
                        help="Which mode to use")

    return parser.parse_args()


class Chexpert(Dataset):
    """Chexpert dataset"""
    
    def __init__(self, split_dir, data_dir, mode='train'):
        """
        Args:
            root_dir (string): Directory with all images.
        """
                    
        self.trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self.data_dir = data_dir
        
        with open(split_dir + '/CheXpert-v1.0-small/' + mode + '_labels.p', 'rb') as handle:
            self.labels = pickle.load(handle)
        with open(split_dir + '/CheXpert-v1.0-small/' + mode + '_imgs.p', 'rb') as handle:
            self.imgs = pickle.load(handle)
        
        self.labels_dict = {'No Finding':0, 'Enlarged Cardiomediastinum': 1, 'Cardiomegaly':2, 'Lung Lesion':3, 'Lung Opacity':4, 
                            'Edema':5, 'Consolidation':6, 'Pneumonia':7, 'Atelectasis':8, 'Pneumothorax':9, 'Pleural Effusion':10,
                            'Pleural Other':11, 'Fracture':12, 'Support Devices':13}
    
        new_labels = [self.labels_dict[label] for label in self.labels]
        self.labels = new_labels
                      
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(self.data_dir + '/' + img_path).convert('RGB')
        img = self.trans(img)
            
        return img, self.labels[idx]
    
    
class CXR14(Dataset):
    """CXR14 dataset"""
    
    def __init__(self, split_dir='/mnt/dsi_vol1/users/frenkel2/data/calibration/focal_calibration-1/CXR14', mode='train'):
        """
        Args:
            root_dir (string): Directory with all images.
        """
                    
        self.trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        
        with open(split_dir + '/' + mode + '_labels.p', 'rb') as handle:
            self.labels = pickle.load(handle)
        with open(split_dir + '/' + mode + '_imgs.p', 'rb') as handle:
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
        
        return img, self.labels[idx]
    

if __name__ == "__main__":
    
    args = parseArgs()
    
    if args.ds_name == 'chexpert':
        ds = Chexpert(split_dir=args.load_split_path, data_dir=args.load_data_path, mode=args.mode)
    elif args.ds_name == 'cxr14':
        ds = CXR14(mode=args.mode)
    