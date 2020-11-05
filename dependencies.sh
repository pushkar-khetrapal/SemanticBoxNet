pip install torch==1.4.0 torchvision==0.5.0 pycocotools numpy==1.16.0 opencv-python tqdm tensorboard tensorboardX pyyaml webcolors matplotlib
cd SemanticBoxNet

mkdir weights
wget https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d1.pth -O weights/efficientdet-d1.pth

