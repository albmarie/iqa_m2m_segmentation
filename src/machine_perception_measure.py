from PIL import Image
import numpy as np
import cv2 as cv
import torch
from class_utils import Machine_Perception_mode
from torchmetrics import JaccardIndex as intersect_and_union

####################################################################################################
####################################################################################################
####################################################################################################

CLASSES = ('road', 'sidewalk', 'building', 'wall',
                'fence', 'pole', 'traffic light', 'traffic sign',
                'vegetation', 'terrain', 'sky', 'person',
                'rider', 'car', 'truck', 'bus',
                'train', 'motorcycle', 'bicycle')

PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                [0, 80, 100], [0, 0, 230], [119, 11, 32]]

#gray = 0.299*R + 0.587*G + 0.114B
#np.array([90, 120, 70, 108, 164, 153, 178, 195, 119, 210, 118, 84, 76, 16, 8, 47, 58, 26, 46])
PALETTE_grayscale = cv.cvtColor(np.array([PALETTE], dtype=np.uint8), cv.COLOR_RGB2GRAY)
IoU_fn = intersect_and_union(task='multiclass', num_classes=len(CLASSES), average='micro')

####################################################################################################
##########          FUNCTIONS          #############################################################
####################################################################################################

"""
Compute pixel-wise accuracy between segmentation prediction on distorted image distorted_prediction_img and pseudo ground truth pseudo_ground_truth_img.
Function can be used for both image-level and block-level experiment, depending on the provided distorted_prediction_img and pseudo_ground_truth_img (which could be pillow object of whole image or blocks).
"""
def compute_acc(distorted_prediction_img, pseudo_ground_truth_img):
    distorted_prediction_img = distorted_prediction_img.convert('L')
    pseudo_ground_truth_img = pseudo_ground_truth_img.convert('L')

    img_size = distorted_prediction_img.size
    acc = (img_size[0]*img_size[1] - len(np.nonzero(np.array(pseudo_ground_truth_img, dtype=np.float32) != np.array(distorted_prediction_img, dtype=np.float32))[0]))/(img_size[0]*img_size[1])
    return acc

####################################################################################################

def compute_mIoU(distorted_prediction_img, pseudo_ground_truth_img):
    #Convert RGB values to gray (with the colors in self.PALETTE, it works because RGB->Y is a bijective function)
    img_np_prediction = np.array(distorted_prediction_img)
    img_np_ground_truth = np.array(pseudo_ground_truth_img)
    img_np_prediction_gray = cv.cvtColor(img_np_prediction, cv.COLOR_RGB2GRAY)
    img_np_ground_truth_gray = cv.cvtColor(img_np_ground_truth, cv.COLOR_RGB2GRAY)
        
    distorted_prediction_classes, pseudo_ground_truth_classes = np.empty(img_np_prediction_gray.shape, dtype=np.uint8), np.empty(img_np_ground_truth_gray.shape, dtype=np.uint8)
    for class_idx in range(len(CLASSES)):
        distorted_prediction_classes[img_np_prediction_gray == PALETTE_grayscale[0,class_idx]] = class_idx
        pseudo_ground_truth_classes[img_np_ground_truth_gray == PALETTE_grayscale[0,class_idx]] = class_idx
    distorted_prediction_classes, pseudo_ground_truth_classes = torch.from_numpy(distorted_prediction_classes), torch.from_numpy(pseudo_ground_truth_classes)
    
    mIoU_score = IoU_fn(distorted_prediction_classes, pseudo_ground_truth_classes)
    return mIoU_score

####################################################################################################

def get_machine_perception_fn(machine_perception: Machine_Perception_mode):
    if machine_perception == Machine_Perception_mode.acc:
        return lambda distorted_prediction_img, pseudo_ground_truth_img: compute_acc(distorted_prediction_img, pseudo_ground_truth_img)
    elif machine_perception == Machine_Perception_mode.mIoU:
        return lambda distorted_prediction_img, pseudo_ground_truth_img: compute_mIoU(distorted_prediction_img, pseudo_ground_truth_img)
    else:
        raise Exception(f'Unsupported machine perception mode ({machine_perception}).')

####################################################################################################
##########          MACHINE PERCEPTION          ####################################################
####################################################################################################

class Machine_Perception_fn(object):
    def __init__(self, machine_perception: Machine_Perception_mode):
        self.machine_perception = machine_perception
        self.machine_perception_fn = get_machine_perception_fn(self.machine_perception)

    def __call__(self, distorted_prediction_img_path, pseudo_ground_truth_img_path, CU_coords, CU_size):
        if CU_coords is None or CU_size is None:
            if CU_coords is not None or CU_size is not None:
                raise Exception(f'CU_coords and CU_size should be either not provided (None as default), or both set to a given value. CU_coords={CU_coords}, CU_size={CU_size}')

        distorted_prediction_img = Image.open(distorted_prediction_img_path)
        pseudo_ground_truth_img = Image.open(pseudo_ground_truth_img_path)

        if CU_coords is not None: #If machine perception measure computed on block-level, then extract corresponding block
            distorted_prediction_img = distorted_prediction_img.crop((CU_coords[0], CU_coords[1], CU_coords[0]+CU_size[0], CU_coords[1]+CU_size[1]))
            pseudo_ground_truth_img = pseudo_ground_truth_img.crop((CU_coords[0], CU_coords[1], CU_coords[0]+CU_size[0], CU_coords[1]+CU_size[1]))

        return self.machine_perception_fn(distorted_prediction_img,pseudo_ground_truth_img)

####################################################################################################
####################################################################################################
####################################################################################################
