from __future__ import division
from .cfg import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from .util import *
import argparse
import os
import os.path as osp
from .darknet import Darknet
from .preprocess import prep_image, inp_to_image
import pandas as pd
import random
import pickle as pkl
import itertools

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
colors = pkl.load(open(os.path.join(THIS_FOLDER, "pallete"), "rb"))
classes = load_classes(os.path.join(THIS_FOLDER, 'cfg/classes.txt'))
model = Darknet(os.path.join(THIS_FOLDER, 'cfg/my-test.cfg'))
model.load_weights(os.path.join(THIS_FOLDER, 'yolov3-voc_2000.weights'))
model.eval()

def arg_parse():
   """
    Parse arguements to the detect module

   """

   parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   parser.add_argument("--confidence", dest = "confidence", help = "Object
   Confidence to filter predictions", default = 0.35)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS
   Threshhold", default = 0.4)
   parser.add_argument("--reso", dest = 'reso', help =
                       "Input resolution of the network. Increase to increase
   accuracy. Decrease to increase speed",
                       default = "480", type = str)

   return parser.parse_args()

def write_rect(output, img):
    c1 = tuple(output[1:3].int())
    c2 = tuple(output[3:5].int())
   label = "{0}".format(classes[int(output[-1])])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
   t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
   c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
   cv2.rectangle(img, c1, c2,color, -1)
   cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
   cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
   return img

def run(img):
   args = arg_parse()
   print(torch.__version__)
   confidence = float(args.confidence)
   nms_thesh = float(args.nms_thresh)
   num_classes = 2

   model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
   assert inp_dim % 32==0
   assert inp_dim > 32    #Detection phase
   batch, orig_img, im_dim = prep_image(img, inp_dim)
    im_dim_list = torch.FloatTensor([im_dim]).repeat(1,2)
    with torch.no_grad():
        prediction = model(Variable(batch), False)
   output = write_results(prediction, confidence, num_classes, nms = True,
   nms_conf = nms_thesh)
   im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
   scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
  output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
  output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
  output[:,1:5] /= scaling_factor

    for i in range(output.shape[0]):
   output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
  output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1]) 
   [write_rect(x, img) for x in output]
    class_name_grouped_dict = { name: [] for name in classes }

   [ class_name_grouped_dict[classes[int(x[-1])]].append([x[1:3].int().numpy(),
   x[3:5].int().numpy()]) for x in output]
for k in list(class_name_grouped_dict.keys()):
    if len(class_name_grouped_dict[k]) == 0:
        del class_name_grouped_dict[k]
return img, class_name_grouped_dict
