import os
import sys
import time
import torch
from torch.backends import cudnn
from matplotlib import colors
from backbone import EfficientDet_semanticBackbone, EfficientDetBackbone
import cv2
import numpy as np
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def box(preds, imgs, color_list, obj_list, imshow=False, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


    return imgs[0]

def inference():
    compound_coef = 0
    force_input_size = None  # set None to use default size
    img_path = 'test/original_img.jpg'
    
    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    threshold = 0.2
    iou_threshold = 0.2

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']


    color_list = standard_to_bgr(STANDARD_COLORS)
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[2] if force_input_size is None else force_input_size
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    model = EfficientDet_semanticBackbone(compound_coef=1, num_classes=len(obj_list),
                                ratios=anchor_ratios, scales=anchor_scales)

    model.load_state_dict(torch.load('model_weight/model_1_epoch_80.pth'))

    if use_cuda:
        model = model.cuda()

    with torch.no_grad():
        features, regression, classification, anchors, sem_out = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

    out = invert_affine(framed_metas, out)
    out = box(out, ori_imgs, color_list, obj_list, imshow=False, imwrite=False)

    outputs = sem_out.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
    pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
    pred_label_imgs = pred_label_imgs.astype(np.uint8)

    z = cv2.resize(pred_label_imgs[0], (ori_imgs[0].shape[1], ori_imgs[0].shape[0]))

    from semantic_utils.utils import label_img_to_color
    pred_label_img_color = label_img_to_color(z)
    overlayed_img = 0.35*out + 0.65*pred_label_img_color

    flag = cv2.imwrite('test/semantic_img_1.jpg', overlayed_img)
    return flag
if __name__ == "__main__":
    flag = inference()
    if flag:
        print("Output Image is saved!")
    else:
        print("Error!")