from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt

import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints



# We feed frames from the webcam into a detector, 
# then we estimate the pose for each detected people in the frame.

ctx = mx.cpu()

detector = get_model("ssd_512_mobilenet1.0_coco", pretrained=True, ctx=ctx)

# To speed up the detector, we can reset the prediction head to only include the classes we need.

detector.reset_class(classes=['person'], reuse_weights={'person':'person'})

# Next for the estimator, we choose ``simple_pose_resnet18_v1b`` for it is light-weighted.
# The default ``simple_pose_resnet18_v1b`` model was trained with input size 256x192.
# We also provide an optional ``simple_pose_resnet18_v1b`` model trained with input size 128x96.
# The latter one is going to be faster, which means a smoother webcam demo.
# Remember that we can load an optional pre-trained model by passing its shasum to ``pretrained``.

estimator = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)

# Acessing webcam.
#cap = cv2.VideoCapture('soccer_low.mp4')
cap = cv2.VideoCapture(0)
time.sleep(1)  ### letting the camera autofocus

# Estimation loop 
# --------------
# For each frame, we perform the following steps:
#
# - loading the webcam frame
# - pre-process the image
# - detect people in the image
# - post-process the detected people
# - estimate the pose for each person
# - plot the result

axes = None

while(True):
    ret, frame = cap.read()

    # Fixing the mirroring of the current frame
    frame = cv2.flip(frame,1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

    x, frame = gcv.data.transforms.presets.yolo.transform_test(frame, short=512, max_size=350)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs,
                                                        output_shape=(128, 96), ctx=ctx)
    if len(upscale_bbox) > 0:
        predicted_heatmap = estimator(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

        img = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores, 
                                box_thresh=1.99, keypoint_thresh=0.2)
    print(bounding_boxs)
    cv_plot_image(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Releasing the webcam before exiting:

cap.release()
