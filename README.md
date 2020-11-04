# SemanticBoxNet

The SemanticBoxNet is a Deep Learning Model which predicts semantic mask and bounding boxes in a single Network. The Architecture contains two heads over a single backbone. The aim of this algorithm is to improve Panoptic Segmentation for realtime. Usually total parameters of Panoptic Segmentation is about 50-70 Million. This algorithm contains 12.5 Million parameters. For inference, since the size of this Architecture is small we can increase batch size to 8 or 16 to increase the speed in same memory usage.

## How to Use

![](media/architecture.jpg)

This project is extension of [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch). The EfficientNet-B1 used as a Backbone. The bounding box head is adpoted from EfficientDet.
The proposed semantic segmentation head consists of three components, each aimed at targeting one of the critical requirements. First, at large-scale, the network should have the ability to capture fine features efficiently. In order to enable this, we employ our Large Scale Feature Extractor (LSFE) module that has four 3×3 separable convolutions with 256 output filters, each followed by swish activation function. The first 3×3 separable convolution increase the number of filters to 256 and the second to fourth 3×3 separable convolution further learns deeper features. The second requirement is that at small-scale, the network should be able to capture long-range context. We employ a modified DPC module in our semantic head. The DPC module consists of a 3×3 separable convolution with 256 output channels having a dilation rate of (1, 6) and extends out to five parallel branches. Three of the branches, each consist of a 3 × 3 dilated separable convolution with 256 outputs, where the dilation rates are (1, 6), (1, 1), and (6, 12) respectively. The fourth branch takes the output of the dilated separable convolution with a dilation rate of (15, 12), as input and passes it through another 3×3 dilated separable convolution with 256 output channels and a dilation rate of (6,3). The outputs from all these parallel branches are then concatenated to yield a tensor with 1280 channels. This tensor is then finally passed through a 1×1 convolution with 256 output channels and forms the output of the DPC module. The third and final requirement for the semantic head is that it should be able to mitigate the mismatch between largescale and small-scale features while performing feature aggregation. To this end, we employ our Mismatch Correction Module (MC) that correlates the small-scale features with respect to large-scale features. It consists of five cascaded 3×3 separable convolutions with 256 output channels, followed by swish activation function and a bilinear upsampling layer that upsamples the feature maps by a factor of 2. After every block the bilinear upsampling is used to upsample the size by factor of 2. As shown in figure below, all the output from DPC and LSPE were concatenated into a single big array. Further passing through set of convolutional layers then upsample it by 2. Last convolutional layer have 20 output channels for CityScapes classes. This architecture is design for 1024x2048 frames.



## Dataset details

DDAD includes high-resolution, long-range [Luminar-H2](https://www.luminartech.com/technology) as the LiDAR sensors used to generate pointclouds, with a maximum range of 250m and sub-1cm range precision. Additionally, it contains six calibrated cameras time-synchronized at 10 Hz, that together produce a 360 degree coverage around the vehicle. The six cameras are 2.4MP (1936 x 1216), global-shutter, and oriented at 60 degree intervals. They are synchronized with 10 Hz scans from our Luminar-H2 sensors oriented at 90 degree intervals (datum names: `camera_01`, `camera_05`, `camera_06`, `camera_07`, `camera_08` and `camera_09`) - the camera intrinsics can be accessed with `datum['intrinsics']`. The data from the Luminar sensors is aggregated into a 360 point cloud covering the scene (datum name: `lidar`). Each sensor has associated extrinsics mapping it to a common vehicle frame of reference (`datum['extrinsics']`).

The training and validation scenes are 5 or 10 seconds long and consist of 50 or 100 samples with corresponding Luminar-H2 pointcloud and six image frames including intrinsic and extrinsic calibration. The training set contains 150 scenes with a total of 12650 individual samples (75900 RGB images), and the validation set contains 50 scenes with a total of 3950 samples (23700 RGB images).

The test set contains 235 scenes, each 1.1 seconds long and consisting of 11 frames, for a total of 2585 frames (15510 RGB images). The middle frame of each validation and each test scene has associated panoptic segmentation labels (i.e. semantic and instance segmentation). These annotations will be used to compute finer gained depth metrics (per semantic class and per instance). Please note that the test annotations **will not be made public**, but will be used to populate the leaderboard on an external evaluation server (coming soon).

<p float="left">
  <img src="/media/figs/pano1.png" width="32%" />
  <img src="/media/figs/pano2.png" width="32%" />
  <img src="/media/figs/pano3.png" width="32%" />
</p>
<img src="/media/figs/odaiba_viz_rgb.jpg" width="96%">
<img src="/media/figs/hq_viz_rgb.jpg" width="96%">
<img src="/media/figs/ann_viz_rgb.jpg" width="96%">

## Dataset stats

### Training split

| Location      | Num Scenes (50 frames)     |  Num Scenes (100 frames)  | Total frames |
| ------------- |:-------------:|:-------------:|:-------------:|
| SF            | 0  |  19 | 1900 |
| ANN           | 23  | 53 | 6450 |
| DET           |  8  | 0  | 400 |
| Japan         | 16  | 31  | 3900 |

Total: `150 scenes` and `12650 frames`.

### Validation split

| Location      | Num Scenes (50 frames)     |  Num Scenes (100 frames)  | Total frames |
| ------------- |:-------------:|:-------------:|:-------------:|
| SF            | 1  |  10 | 1050 |
| ANN           | 11  | 14 | 1950 |
| Japan         | 9  | 5  | 950 |

Total: `50 scenes` and `3950 frames`.


### Test split

| Location      | Num Scenes (11 frames)      | Total frames |
| ------------- |:-------------:|:-------------:|
| SF            | 69  | 759  |
| ANN           | 49  | 539  |
| CAM           | 61  | 671  |
| Japan         | 56  | 616  |

Total: `235 scenes` and `2585 frames`.

USA locations: ANN - Ann Arbor, MI; SF - San Francisco Bay Area, CA; DET - Detroit, MI; CAM - Cambridge, Massachusetts. Japan locations: Tokyo and Odaiba.

## Sensor placement

The figure below shows the placement of the DDAD LiDARs and cameras. Please note that both LiDAR and camera sensors are positioned so as to provide 360 degree coverage around the vehicle. The data from all sensors is time synchronized and reported at a frequency of 10 Hz. The data from the Luminar sensors is reported as a single point cloud in the vehicle frame of reference with origin on the ground below the center of the vehicle rear axle, as shown below. For instructions on visualizing the camera images and the point clouds please refer to this [IPython notebook](media/notebooks/DDAD.ipynb).

![](media/figs/ddad_sensors.png)

## Evaluation metrics

Please refer to the the [Packnet-SfM](https://github.com/TRI-ML/packnet-sfm) codebase for instructions on how to compute detailed depth evaluation metrics.

## IPython notebook

The associated [IPython notebook](notebooks/DDAD.ipynb) provides a detailed description of how to instantiate the dataset with various options, including loading frames with context, visualizing rgb and depth images for various cameras, and displaying the lidar point cloud.

[![](media/figs/notebook.png)](notebooks/DDAD.ipynb)

## References

Please use the following citation when referencing DDAD:

#### 3D Packing for Self-Supervised Monocular Depth Estimation (CVPR 2020 oral)
*Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos and Adrien Gaidon*, [**[paper]**](https://arxiv.org/abs/1905.02693), [**[video]**](https://www.youtube.com/watch?v=b62iDkLgGSI)
```
@inproceedings{packnet,
  author = {Vitor Guizilini and Rares Ambrus and Sudeep Pillai and Allan Raventos and Adrien Gaidon},
  title = {3D Packing for Self-Supervised Monocular Depth Estimation},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  primaryClass = {cs.CV}
  year = {2020},
}
```


## Privacy

To ensure privacy the DDAD dataset has been anonymized (license plate and face blurring) using state-of-the-art object detectors.


## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
