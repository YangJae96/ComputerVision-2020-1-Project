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
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="2,3"


parser = argparse.ArgumentParser(description='CVProject')
parser.add_argument('--datapath', default='/home/yjw/DeepLearning/Computer-Vision-2020-1/',
                    help='default path')
parser.add_argument('--loadmodel', default='/home/yjw/DeepLearning/Computer-Vision-2020-1/trained_model/cvd.pth',
                    help='loading model')
args = parser.parse_args()
path=args.datapath

USE_GPU = True

dtype = torch.float32 

if torch.cuda.is_available():
    device=torch.device('cuda')

print_every = 100
print('using device:', device)
print(torch.cuda.get_device_name())

simple_transform=transforms.Compose([transforms.Resize((224,224))
                                    ,transforms.ToTensor()
                                    ,transforms.Normalize([0.485,0.456,
    0.406], [0.229, 0.224, 0.225])])


test=ImageFolder(path+ 'my_dataset',simple_transform)
test_data_gen=torch.utils.data.DataLoader(test, batch_size=128, num_workers=3)

def create_model():
	model_ft = models.resnet18(pretrained=True)
	num_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_ftrs, 4)
	pytorch_total_params = sum(p.numel() for p in model_ft.parameters())
	return model_ft

def validation(model, testloader, criterion):
    model = model.to(device=device, dtype=dtype)
    model.train(False)  # ?? ?? ??
    
    run_loss = 0
    run_corrects = 0

    for inputs, labels in testloader:
    
        if torch.cuda.is_available():
            inputs = inputs.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=torch.long)
        # forward

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)

        run_loss += loss.data
        run_corrects += torch.sum(preds == labels.data)

        epoch_loss = run_loss.item() / len(testloader.dataset)
        epoch_acc = run_corrects.item() / len(testloader.dataset)

    return epoch_loss, epoch_acc  

if __name__ == '__main__':

	model = create_model()
	model.load_state_dict(torch.load(args.loadmodel))
	# model.load_state_dict(torch.load(path+'trained_model/cvd.pth'))
	model = nn.DataParallel(model)
	model.to(device=device)

	learning_rate = 0.001
	criterion = nn.CrossEntropyLoss()
	optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
	start = time.time()
	_,acc=validation(model, test_data_gen, criterion)
	print("Model Accuracy == {}% ".format(acc*100))
	print("Test Time == {}".format(time.time()-start))

	

	