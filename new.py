

from PIL import Image	
from imutils import paths

from tqdm import tqdm
import numpy as np
import cv2
import gc
import numpy as np

import datetime, os

from matplotlib import image
from matplotlib import pyplot
import matplotlib.image as mpimg
x_path = []
rootdir = 'C:/Users/Kashif/Desktop/Flask Segmentation/Img'
for root, dirs, files in os.walk(rootdir):
	for name in files:
		if name.endswith((".jpg")):
			x_path.append(os.path.join(root, name).replace('\\','/'))
	
	###########################
print(x_path)
pathOut = video_name = 'Output_.avi'
frame_array = []
files = x_path
 
    #for sorting the file names properly
files.sort()
 
for i in range(len(files)):
    filename=files[i]
        #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
        
        #inserting the frames into an image array
    frame_array.append(img)
 
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for i in range(len(frame_array)):
        # writing to a image array
    out.write(frame_array[i])
out.release()