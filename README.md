# Computer-Vision-2020-1
# 2020-1st Computer Vision Project 
2020학년도 1학기 컴퓨터비전 기말프로젝트로 Weapon Classifier를 만들기를 위한 레포지토리입니다.

## 1. Contributor
- 양재원, 유재원, 부준영

## 2. Version
- python3.6
- torch1.5 and torchvision 0.6
- Cuda >=10.2 
- Ubuntu 18.04
- Multiple GPU: Nvidia Titan X

## 3. Train
- Must to do: You need to unzip the dataset.zip into the project root dir. The dataset folder should be in same dir as train.py 
- command: python train.py --datapath (Root Dir of this Project) --epoch (epochs to train) --pretrained (set True for Pretranied ResNet Model)

## 4. Test
- Command: python test.py --datapath (Root Dir of this Project)

## 5. Classify your Weapon(knife,axe,pistol,rifle)
- Commnd: python demo.py
- Put your weapon data in the my_dataset dir and put it in each subfolder.
 It will classifiy the data with the trained model  "cvd.pth" inside the "trained_model" dir. 

