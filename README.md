# SemanticBoxNet

The SemanticBoxNet is a Deep Learning Model which predicts semantic mask and bounding boxes in a single Network. The Architecture contains two heads over a single backbone. The aim of this algorithm is to improve Panoptic Segmentation for realtime. Usually total parameters of Panoptic Segmentation is about 50-70 Million. This algorithm contains 12.5 Million parameters. For inference, since the size of this Architecture is small we can increase batch size to 8 or 16 to increase the speed in same memory usage.

## How to Use

![](/media/architecture_sem.jpg)

This project is extension of [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch). The EfficientNet-B1 used as a Backbone. The bounding box head is adpoted from EfficientDet.
The proposed semantic segmentation head consists of three components, each aimed at targeting one of the critical requirements. First, at large-scale, the network should have the ability to capture fine features efficiently. The second requirement is that at small-scale, the network should be able to capture long-range context. We employ a modified DPC module in our semantic head. The third and final requirement for the semantic head is that it should be able to mitigate the mismatch between largescale and small-scale features while performing feature aggregation. To this end, we employ our Mismatch Correction Module (MC) that correlates the small-scale features with respect to large-scale features. After every block the bilinear upsampling is used to upsample the size by factor of 2. As shown in figure below, all the output from DPC and LSPE were concatenated into a single big array. Further passing through set of convolutional layers then upsample it by 2. Last convolutional layer have 20 output channels for CityScapes classes. This architecture is design for 1024x2048 frames.

![](/media/sem_ach.jpg)


<!-- 
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
 -->