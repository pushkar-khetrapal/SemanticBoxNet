# SemanticBoxNet

The SemanticBoxNet is a Deep Learning Model which predicts semantic mask and bounding boxes in a single Network. The Architecture contains two heads over a single backbone. The aim of this algorithm is to improve Panoptic Segmentation for realtime. Usually total parameters of Panoptic Segmentation is about 50-70 Million. This algorithm contains 12.5 Million parameters. For inference, since the size of this Architecture is small we can increase batch size to 8 or 16 to increase the speed in same memory usage.

## Architecture

![](/media/architecture_sem.jpg)

This project is extension of [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch). The EfficientNet-B1 used as a Backbone. The bounding box head is adpoted from EfficientDet.
The proposed [semantic segmentation head](http://panoptic.cs.uni-freiburg.de/) consists of three components, each aimed at targeting one of the critical requirements. First, at large-scale, the network should have the ability to capture fine features efficiently. The second requirement is that at small-scale, the network should be able to capture long-range context. We employ a modified DPC module in our semantic head. The third and final requirement for the semantic head is that it should be able to mitigate the mismatch between largescale and small-scale features while performing feature aggregation. To this end, we employ our Mismatch Correction Module (MC) that correlates the small-scale features with respect to large-scale features. After every block the bilinear upsampling is used to upsample the size by factor of 2. As shown in figure below, all the output from DPC and LSPE were concatenated into a single big array. Further passing through set of convolutional layers then upsample it by 2. Last convolutional layer have 20 output channels for CityScapes classes. This architecture is design for 1024x2048 frames.

![](/media/sem_arch.jpg)

## To Do List

1. Training about 300 epoches.
2. Generate Instance mask from semantic segmentation and bounding boxes.
3. Panoptic Fusion.
4. Optimizing it further with Nvidia TensorRT.

## Training
I used Pre-trained weights from EfficientDet algorithm and train semantic head only. I trained it for 80 epoches, about 53 hours on Google Colab(one, k-80 GPU) on CityScapes dataset till now. 

## Accuracy
### Bounding Boxes (Results are same as in EfficientDet-d1)

```
efficientdet-d1
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.388
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.577
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.413
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.189
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.441
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.564
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.312
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.480
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.513
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.287
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.580
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.699
```
### Semantic Output
Current accuracy of semantic is 66.1 mIoU. The accuracy is still improving need long training to get good results. 
## Outputs

<p float="left">
  <img src="/media/original_img.jpg" width="48%" />
  <img src="/media/semantic_img_12.jpg" width="48%">
  <img src="/media/original_img1.jpg" width="48%" />
  <img src="/media/semantic_img_13.jpg" width="48%">
  <img src="/media/original_img2.jpg" width="48%" />
  <img src="/media/semantic_img_14.jpg" width="48%">
  <img src="/media/original_img3.jpg" width="48%" />
  <img src="/media/semantic_img_15.jpg" width="48%">
</p>

## How to use
### Training an Inference
```
To train the network, you need to download CityScapes dataset (~10GB).
git clone https://github.com/pushkar-khetrapal/SemanticBoxNet.git
sh SemanticBoxNet/dependencies.sh

To download the dataset, you just need to update CityScapes credentials in place of ####.
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

python train_sembox.py


```
### Only Inference

```
git clone https://github.com/pushkar-khetrapal/SemanticBoxNet.git
sh SemanticBoxNet/dependencies.sh
python sembox_inference.py

```
##### See output at test/semantic_img{}.jpg

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.