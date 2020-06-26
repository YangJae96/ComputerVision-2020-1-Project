from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import time
import os
import cv2 as cv
from model import AlexNet
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3"
parser = argparse.ArgumentParser(description='CVProject')
parser.add_argument('--datapath', default='/home/yjw/DeepLearning/Computer-Vision-2020-1/',
                    help='default path')
parser.add_argument('--pretrained', default=False,
                    help='default path')
parser.add_argument('--epoch', default=10,
                    help='train epoch')
args = parser.parse_args()

path=args.datapath

USE_GPU = True
dtype = torch.float32 

if torch.cuda.is_available():
    device=torch.device('cuda')

print_every = 100
print('using device:', device)
print(torch.cuda.get_device_name())

no_of_images=25000
shuffle=np.random.permutation(no_of_images)

simple_transform=transforms.Compose([transforms.Resize((224,224))
                                    ,transforms.ToTensor()
                                    ,transforms.Normalize([0.485,0.456,
    0.406], [0.229, 0.224, 0.225])])
train=ImageFolder(path+'dataset/train/',simple_transform)
valid=ImageFolder(path+'dataset/valid/',simple_transform)

print(train.class_to_idx)
print(train.classes)

train_data_gen = torch.utils.data.DataLoader(train,shuffle=True,batch_size=128  ,num_workers=3)
valid_data_gen = torch.utils.data.DataLoader(valid,batch_size=128,num_workers=3)    
dataset_sizes = {'train':len(train_data_gen.dataset),
                 'valid':len(valid_data_gen.dataset),}
dataloaders = {'train':train_data_gen,'valid':valid_data_gen}

def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    cv.imshow('test',inp)
    cv.waitKey(0)
    cv.destroyAllWindows()


def create_model():
    if args.pretrained:
        model_ft = models.resnet18(pretrained=True)
    elif args.pretrained==False:
        model_ft = models.resnet18()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 4)
    pytorch_total_params = sum(p.numel() for p in model_ft.parameters())
    return model_ft

def train_model(model, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True) #Set Train Mode
            else:
                model.train(False) #Set Valid Mode
            
            running_loss = 0.0
            running_corrects=0
        
            for data in dataloaders[phase]:
                x, y = data
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)
                
                optimizer.zero_grad()
                scores = model(x)
                _, preds = torch.max(scores.data, 1)
                loss = criterion(scores, y)

                if phase =='train':
                    loss.backward()
                    optimizer.step()

                running_loss+=loss.data
                running_corrects+=torch.sum(preds==y.data)
        
            epoch_loss = running_loss.item() / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.module.state_dict()
                
        time_elapsed = time.time() - since
        print('Training on way in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        
        # if epoch+1 ==num_epochs and epoch is not 0:
        # 	print("Hello")
        # 	torch.save(best_model_wts,"cvd.pt")
        # 	break
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
      
    return model

if __name__ == '__main__':
    model = create_model()
    model = nn.DataParallel(model)
    model.to(device=device)

    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_trained = train_model(model, optimizer_ft, 
                       exp_lr_scheduler, num_epochs=int(args.epoch))

    torch.save(model_trained.module.state_dict(),path+"trained_model/cvd.pth")



# imshow(train[5000][0])