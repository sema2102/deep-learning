import torch import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import os
import numpy as np
from skimage import segmentation
import torch.nn.init
import collections
debug = False
minLabels = 4 # Minimum number of class
maxIter = 50
nChannel = 50 # Number of output channel
green_color = (0, 255, 0)
fruit_color = (255, 0, 0)
flower_color = (255, 255, 0)
back_color = (160, 82, 45)
color_pattern = { 'fruit': fruit_color, 'folower': flower_color, 'green':
   green_color }
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

# CNN model
class SegmentationNet(nn.Module):
    def __init__(self, input_dim, nChannel):
        super(SegmentationNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel,\
                                 kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = []
        self.bn2 = []
        for i in range(1):
            self.conv2.append( nn.Conv2d(nChannel, nChannel,\
                                    kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(nChannel) )
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1,\
                                                  stride=1, padding=0 )
       self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
 x= self.conv1(x)
 x= F.relu( x )
 x= self.bn1(x)
       for i in range(len(self.conv2) - 1):
x = self.conv2[i](x)
x=F.relu(x)
x = self.bn2[i](x)
x= self.conv3(x)
x= self.bn3(x)
return x

x = self.conv2[i](x) x=F.relu(x)
x = self.bn2[i](x) self.conv3(x) self.bn3(x)
def calc_ratio(im_rgb):
   colors, count = np.unique(np.reshape(im_rgb, (im_rgb.shape[0] *
   im_rgb.shape[1], 3)), axis=0, return_counts = True)
   rate_dict = { key: np.where((colors == color).all(axis=1))[0] for key,
   color in color_pattern.items() }
   rate_dict = { key: (count[val[0]] if len(val) != 0 else 0) for key, val in
   rate_dict.items() }
    return rate_dict

# img:
# n_segment:
# crop_rects:
# is_mask:
def run(img, n_segment, crop_rects, is_mask = False):
    data = torch.from_numpy( np.array([img.transpose(\
                                  (2, 0, 1) ).astype('float32')/255.]) )
    data = Variable(data)
    labels = segmentation.slic(img, compactness=10, n_segments=n_segment)
    labels = labels.reshape(img.shape[0]*img.shape[1])
   u_labels = np.unique(labels)
   l_inds = [] # stored property  for indexes grouped by super pixel index
   for i in range(len(u_labels)):
 l_inds.append( np.where( labels == u_labels[ i ])[0])

 # train
    model = SegmentationNet( data.size(1), nChannel )
 # use
    check_point = "./saved_model_for_mask.pth" if is_mask else "./
   saved_model.pth"
    model.load_state_dict(torch.load(os.path.join(THIS_FOLDER, check_point),
   map_location="cpu"))
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    label_colours = np.random.randint(255,size=(100,3))
    for batch_idx in range(maxIter):
        # forwarding
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
        _, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        for i in range(len(l_inds)):
            labels_per_sp = im_target[ l_inds[ i ] ]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
hist[ j ] = len( np.where( \
 labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
 im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
 target = torch.from_numpy( im_target )
 target = Variable( target )
loss = loss_fn(output, target)
 loss.backward()
 optimizer.step()
 
    # debug
    if debug:
        print (batch_idx, '/', maxIter, ':', nLabels, loss.data[0])
        if nLabels <= minLabels:
            print ("nLabels", nLabels, "reached minLabels",\
                                                       minLabels, ".")
break
if is_mask:
    mask_arr = np.zeros( img.shape )
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
    ignore, target = torch.max( output, 1 )
    target = target.data.cpu().numpy()
    count = np.bincount(target)
    mode = np.argmax(count)
    th = np.where(target == mode, 255,0)\
           .reshape( img.shape[0:2] ).astype( np.uint8 )
    contours, _ =
cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contour_sizes = [(cv2.contourArea(contour), contour) for contour in
contours]

    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    image = cv2.drawContours(mask_arr,[biggest_contour], 0, (255,0,0),
    return (image[:, :, (0)] == 255)
else:
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    most_common_group = collections.Counter(im_target)\
                                   .most_common()[0][0]
    im_target_rgb = np.array([green_color \
        if most_common_group == c else back_color for c in im_target])
    im_target_rgb = im_target_rgb.reshape( img.shape ).astype( np.uint8 )
    for key, rects in crop_rects.items():
   flower_color
for
rect in rects:
cropped_img = img[rect[0][1]:rect[1][1],\
                  rect[0][0]:rect[1][0]]
target_color = fruit_color if key == "fruit" else
                    im_target_rgb[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
   [run(cv2.GaussianBlur(cropped_img,(5,5),0), 50, [], True)] = target_color
return cv2.cvtColor(im_target_rgb, cv2.COLOR_RGB2BGR), calc_ratio(im_target_rgb)
