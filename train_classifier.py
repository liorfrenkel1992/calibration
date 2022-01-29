import numpy as np
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import resize
from torch.utils.data import DataLoader

import torchxrayvision as xrv

to_pil = transforms.ToPILImage()

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[1:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        
        image = np.transpose(image, (1, 2, 0))

        img = resize(to_pil(image), (new_h, new_w))

        return img

data_dir = 'covid-chestxray-dataset'
def load_split_train_test(datadir, valid_size = .1, test_size = .1, batch_size=16):
    trans = transforms.Compose([Rescale((256, 256)), transforms.ToTensor()])
    data = xrv.datasets.COVID19_Dataset(imgpath=datadir + "/images/",csvpath=datadir + "/metadata.csv", transform=trans)
    # data = xrv.datasets.COVID19_Dataset(imgpath=datadir + "/images/",csvpath=datadir + "/metadata.csv")
    
    len_data = len(data)
    len_val = int(valid_size * len_data)
    len_test = int(test_size * len_data)
    len_train = int(len_data - len_val - len_test)
    train_set, val_set, test_set = torch.utils.data.random_split(data, [len_train, len_val, len_test], generator=torch.Generator().manual_seed(42))
    
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return trainloader, valloader, testloader

lr = 0.003
batch_size = 16

trainloader, valloader, testloader = load_split_train_test(data_dir, .1, .1, batch_size)
print(trainloader.dataset.dataset.labels)

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
model = models.resnet50(pretrained=True)
print(model)

for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, len(trainloader.dataset)),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=lr)
model.to(device)

epochs = 5
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
for epoch in range(epochs):
    for sample in trainloader:
        steps += 1
        inputs, labels = sample["img"], sample["idx"]
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'covid19_resnet50_epochs_{},lr_{},bs_{}.pth'.format(epochs, lr, batch_size))

