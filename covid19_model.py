import os
import argparse

import torch
from torch.nn import functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import models

from sklearn.metrics import confusion_matrix

from dataset import Chexpert, Covid19

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parseArgs():
    load_model_path = 'vanilla_medical_classifier_chexpert'

    parser = argparse.ArgumentParser(
        description="Train/test medical imaging models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--load_model_path", type=str, default=load_model_path, dest="load_model_path",
                        help="Path of saved model")
    parser.add_argument("--model_name", type=str, default="chexpert", dest="model_name",
                        help="model name to train/test")
    parser.add_argument("--n_classes", type=int, default=14, help="number of classes")
    parser.add_argument("-use_sched", action='store_true', help="use scheduler")
    # parser.add_argument("-no_sched", action='store_false', help="don't use scheduler")
    # parser.set_defaults(use_sched=True)
    
    # Chexport args
    parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="training batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="which mode to use")

    return parser.parse_args()

for_pad = lambda s: s if s > 2 else 3

class ConvBlock(nn.Module):
    def __init__(self, ni, nf, size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(ni, nf, kernel_size=size, stride=stride, padding=(for_pad(size) - 1)//2, bias=False),
                                        nn.BatchNorm2d(nf),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True)  
                                        )
        
    def forward(self, x):
        return self.conv_block(x)

class ConvLayer(nn.Module):
    def __init__(self, ni, nf, size=3, stride=1):
        super(ConvLayer, self).__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(ni, nf, kernel_size=size, stride=stride, padding=(for_pad(size) - 1)//2, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm2d(nf),
                                        )
        
    def forward(self, x):
        return self.conv_layer(x)
    
class TripleConv(nn.Module):
    def __init__(self, ni, nf, size=3, stride=1):
        super(TripleConv, self).__init__()
        self.triple_conv = nn.Sequential(ConvBlock(ni, nf),
                                         ConvBlock(nf, ni, size=1),  
                                         ConvBlock(ni, nf)
                                         )
        
    def forward(self, x):
        return self.triple_conv(x)
    
class DarkCovidNet(nn.Module):
    def __init__(self, output_size=3):
        super(DarkCovidNet, self).__init__()
        # self.conv_block1 = ConvBlock(1, 8)
        # self.conv_block2 = ConvBlock(8, 16)
        # self.conv_block3 = ConvBlock(256, 128, size=1)
        # self.conv_block4 = ConvBlock(128, 256)
        # self.triple_conv1 = TripleConv(16, 32)
        # self.triple_conv2 = TripleConv(32, 64)
        # self.triple_conv3 = TripleConv(64, 128)
        # self.triple_conv4 = TripleConv(128, 256)
        # self.maxpool1 = nn.MaxPool2d(2, stride=2)
        # self.maxpool2 = nn.MaxPool2d(2, stride=2)
        # self.maxpool3 = nn.MaxPool2d(2, stride=2)
        # self.maxpool4 = nn.MaxPool2d(2, stride=2)
        # self.maxpool5 = nn.MaxPool2d(2, stride=2)
        # self.conv_layer = ConvLayer(256, 2)
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(338, 3)
        self.layers = nn.Sequential(ConvBlock(1, 8),
                                    nn.MaxPool2d(2, stride=2),
                                    ConvBlock(8, 16),
                                    nn.MaxPool2d(2, stride=2),
                                    TripleConv(16, 32),
                                    nn.MaxPool2d(2, stride=2),
                                    TripleConv(32, 64),
                                    nn.MaxPool2d(2, stride=2),
                                    TripleConv(64, 128),
                                    nn.MaxPool2d(2, stride=2),
                                    TripleConv(128, 256),
                                    ConvBlock(256, 128, size=1),
                                    ConvBlock(128, 256),
                                    ConvLayer(256, 2),
                                    nn.Flatten(),
                                    nn.Linear(338, output_size)
                                    )

    def forward(self, x):
        # x = self.conv_block1(x)
        # x = self.maxpool1(x)
        # x = self.conv_block2(x)
        # x = self.maxpool2(x)
        # x = self.triple_conv1(x)
        # x = self.maxpool3(x)
        # x = self.triple_conv2(x)
        # x = self.maxpool4(x)
        # x = self.triple_conv3(x)
        # x = self.maxpool5(x)
        # x = self.triple_conv4(x)
        # x = self.conv_block3(x)
        # x = self.conv_block4(x)
        # x = self.conv_layer(x)
        # x = self.flatten(x)
        
        # return self.fc(x)
        return self.layers(x)
    

class ResNet(nn.Module):
    def __init__(self, num_classes, is_pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=is_pretrained)
        fc_num_features = self.resnet.fc.in_features
        self.classifier = nn.Linear(in_features=fc_num_features,
                                    out_features=num_classes,
                                    bias=True)
        return

    def forward(self, x):
        res = self.resnet.conv1(x)  # bs x 64 x 112 x 112
        res = self.resnet.bn1(res)  # bs x 64 x 112 x 112
        res = self.resnet.relu(res)  # bs x 64 x 112 x 112
        res = self.resnet.maxpool(res)  # bs x 64 x 56 x 56

        l_1 = self.resnet.layer1(res)  # bs x 256 x 56 x 56
        l_2 = self.resnet.layer2(l_1)  # bs x 512 x 28 x 28
        l_3 = self.resnet.layer3(l_2)  # bs x 1024 x 14 x 14
        res = self.resnet.layer4(l_3)  # bs x 1024 x 7 x 7
        g = self.resnet.avgpool(res)  # bs x 2048 x 1 x 1
        out = self.classifier(g.squeeze())  # bs x num_classes
        return out
    
       
def train(args, model, trainloader, valloader, num_classes=14, batch_size=32, lr=0.0001, epochs=30, print_every=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    #  optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    optimizer = torch.optim.Adam(model.parameters(),
                                          lr=lr,
                                          betas=(0.5, 0.999), amsgrad=True)
    if args.use_sched:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                                        factor=0.8, patience=5, cooldown=3, verbose=True)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    model.to(device)

    steps = 0
    running_loss = 0
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                labels_list = []
                predictions_list = []
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        pred = model(inputs)
                        batch_loss = criterion(pred, labels)
                        test_loss += batch_loss.item()
                        
                        softmaxes = F.softmax(pred, dim=1)
                        confidences, predictions = torch.max(softmaxes, 1)
                        accuracy += torch.mean((predictions.eq(labels)).float()).item()
                        labels_list.extend(labels.cpu().numpy().tolist())
                        predictions_list.extend(predictions.cpu().numpy().tolist())
                        
                conf_mat = confusion_matrix(labels_list, predictions_list)
                train_losses.append(running_loss/len(trainloader))
                val_loss = test_loss/len(valloader)
                test_losses.append(val_loss)                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Confusion matrix: {conf_mat}"
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Val loss: {val_loss:.3f}.. "
                    f"Test accuracy: {accuracy/len(valloader):.3f}")
                running_loss = 0
                
                if args.use_sched:
                    scheduler.step(val_loss)
                model.train()

    if args.use_sched:
        torch.save(model, '{}/{}_resnet50_epochs_{},lr_{},bs_{}_{}_classes_sched.pth'.format(args.load_model_path, args.model_name, epochs, lr, batch_size, num_classes))
    else:
        torch.save(model, '{}/{}_resnet50_epochs_{},lr_{},bs_{}_{}_classes_no_sched.pth'.format(args.load_model_path, args.model_name, epochs, lr, batch_size, num_classes))

def test(args, model, testloader, num_classes=14, batch_size=32, lr=0.0001, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load('{}/{}_resnet50_epochs_{},lr_{},bs_{}_{}_classes.pth'.format(args.load_model_path, args.model_name, epochs, lr, batch_size, num_classes))
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()

    labels_list = []
    predictions_list = []
    running_loss = 0
    count_acc = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            accuracy = 0
            inputs, labels = inputs.to(device), labels.to(device)
            pred = model(inputs)
            loss = criterion(pred, labels)
            running_loss += loss.item()
            
            softmaxes = F.softmax(pred, dim=1)
            _, predictions = torch.max(softmaxes, 1)
            accuracy += torch.mean((predictions.eq(labels)).float()).item()
            count_acc += 1
            labels_list.extend(labels.cpu().numpy().tolist())
            predictions_list.extend(predictions.cpu().numpy().tolist())
                             
        conf_mat = confusion_matrix(labels_list, predictions_list)
                                  
        test_loss = running_loss/len(testloader)
        print(f"Confusion matrix: \n{conf_mat}\n"
            f"Val loss: {test_loss:.3f}.. "
            f"Test accuracy: {accuracy/count_acc:.3f}")                
                
if __name__ == "__main__":
    
    args = parseArgs()
    
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.n_epochs
    n_classes = args.n_classes

    # model = DarkCovidNet(output_size=14)
    model = ResNet(num_classes=n_classes)

    # covid19_ds = Covid19('/mnt/dsi_vol1/users/frenkel2/data/calibration/focal_calibration-1/COVID-19/X-Ray Image DataSet')
    # len_data = len(covid19_ds)
    # len_val = int(0.1 * len_data)
    # len_test = int(0.1 * len_data)
    # len_train = int(len_data - len_val - len_test)
    # train_imgs = covid19_ds.imgs[:len_train]
    # val_imgs = covid19_ds.imgs[len_train:len_train + len_val]
    # test_imgs = covid19_ds.imgs[len_train + len_val:]
    # train_labels = covid19_ds.labels[:len_train]
    # val_labels = covid19_ds.labels[len_train:len_train + len_val]
    # test_labels = covid19_ds.labels[len_train + len_val:]
    # train_set, val_set, test_set = random_split(covid19_ds, [len_train, len_val, len_test], generator=torch.Generator().manual_seed(42))

    # train_set = Covid19('/mnt/dsi_vol1/users/frenkel2/data/calibration/focal_calibration-1/COVID-19', 'train')
    # val_set = Covid19('/mnt/dsi_vol1/users/frenkel2/data/calibration/focal_calibration-1/COVID-19', 'val')
    # test_set = Covid19('/mnt/dsi_vol1/users/frenkel2/data/calibration/focal_calibration-1/COVID-19', 'test')

    if args.mode == 'train':
        train_set = Chexpert(args, type='train')
        val_set = Chexpert(args, type='val')
        trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        valloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
        train(args, model, trainloader, valloader, num_classes=n_classes, epochs=epochs, lr=lr, batch_size=batch_size)
    else:
        test_set = Chexpert(args, type='test')
        testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
        test(args, model, testloader, num_classes=n_classes, epochs=epochs, lr=lr, batch_size=batch_size)
