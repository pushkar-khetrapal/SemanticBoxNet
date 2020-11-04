pip install torch==1.4.0 torchvision==0.5.0 pycocotools numpy==1.16.0 opencv-python tqdm tensorboard tensorboardX pyyaml webcolors matplotlib
cd EfficientDet_Panoptic

mkdir weights
wget https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d1.pth -O weights/efficientdet-d1.pth

## set credentials here
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=####&password=####&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
unzip gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip
mkdir data
mkdir data/cityscapes
mv gtFine data/cityscapes
mv leftImg8bit data/cityscapes
git clone https://github.com/mcordts/cityscapesScripts.git