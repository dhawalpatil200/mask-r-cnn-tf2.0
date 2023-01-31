import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
from mrcnn.config import Config
# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']



# class InferenceConfig(coco.CocoConfig):
class InferenceConfig(mrcnn.config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(class_names)

config = InferenceConfig()
config.display()

# Create model object in inference mode.
# model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model = mrcnn.model.MaskRCNN(mode="inference", config=InferenceConfig(),model_dir=os.getcwd())

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)




IMAGE_SOURCE_DIR = os.path.join(ROOT_DIR, "leopard_images")
OUTPUT_DIR =  os.path.join(ROOT_DIR, "results")


def maskRCNN(image_path):
    
  # Load a random image from the images folder
  image = skimage.io.imread(image_path)
  image_name = 'seg_output_' + image_path.split('/')[-1]
  # print(image_name)

  # original image
  # plt.figure(figsize=(12,10))
  # skimage.io.imshow(image)


  # Run detection
  results = model.detect([image], verbose=1)

  # Visualize results
  r = results[0]
  # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

  mask = r['masks']
  mask = mask.astype(int)
  # print('shape  ',mask.shape)

  for i in range(mask.shape[2]):
    temp = skimage.io.imread(image_path)
    for j in range(temp.shape[2]):
        temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
    cv.imwrite(os.path.join(OUTPUT_DIR,f"{i}_"+image_name),cv.cvtColor(temp, cv.COLOR_BGR2RGB))
    # plt.figure(figsize=(8,8))
    # plt.imshow(temp)
    # print('final path')
    # print(os.path.join(OUTPUT_DIR,image_name))
    



image_name_list = os.listdir(IMAGE_SOURCE_DIR)

for image_name in image_name_list:
  if image_name == 'img_99.jpg':
    image_path = os.path.join(IMAGE_SOURCE_DIR, image_name)
    maskRCNN(image_path)