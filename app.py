#!pip install -U segmentation-models==0.2.1
import pip
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", 'segmentation-models==0.2.1'])
from PIL import Image	
from imutils import paths
from segmentation_models import Unet
from segmentation_models.utils import set_trainable
from tqdm import tqdm
import numpy as np
import cv2
import gc
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
# Load the TensorBoard notebook extension
from keras.callbacks import TensorBoard
import tensorflow as tf
import datetime, os
from keras.callbacks import EarlyStopping
from flask import Flask, jsonify, request
from keras.callbacks import ReduceLROnPlateau
from matplotlib import image
from matplotlib import pyplot
import matplotlib.image as mpimg

def IoU(y_val, y_pred):
    class_iou = []
    n_classes = 8
    
    y_predi = np.argmax(y_pred, axis=3)
    y_truei = np.argmax(y_val, axis=3)
    
    for c in range(n_classes):
        TP = np.sum((y_truei == c) & (y_predi == c))
        FP = np.sum((y_truei != c) & (y_predi == c))
        FN = np.sum((y_truei == c) & (y_predi != c)) 
        IoU = TP / float(TP + FP + FN)
        if(float(TP + FP + FN) == 0):
          IoU=TP/0.001
        class_iou.append(IoU)
    MIoU=sum(class_iou)/n_classes
    return MIoU
def miou( y_true, y_pred ) :
    score = tf.py_function( lambda y_true, y_pred : IoU( y_true, y_pred).astype('float32'),
                        [y_true, y_pred],
                        'float32')
    return score
	
dependencies = {
    'miou': miou
}
from keras.models import load_model
model = load_model('Unet_Resnet.hdf5', custom_objects=dependencies)

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return flask.render_template('index.html')


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
	#os.remove("C:/Users/Kashif/Desktop/Flask Segmentation/data/*.jpg", dir_fd = None)
	#os.remove("C:/Users/Kashif/Desktop/Flask Segmentation/Img/*.jpg", dir_fd = None)
	query = request.form.to_dict()
	print(query ['video'])
	# Read the video from specified path 
	cam = cv2.VideoCapture(query ['video']) 
  
	try: 
      
    # creating a folder named data 
		if not os.path.exists('data'): 
			os.makedirs('data') 
  
# if not created then raise error 
	except OSError: 
		print ('Error: Creating directory of data') 
  
# frame 
	currentframe = 0
  
	while(True): 
      
    # reading from frame 
		ret,frame = cam.read() 
  
		if ret: 
        # if video is still left continue creating images 
			name = './data/frame' + str(currentframe) + '.jpg'
			print ('Creating...' + name) 
  
        # writing the extracted images 
			cv2.imwrite(name, frame) 
  
        # increasing counter so that it will 
        # show how many frames are created 
			currentframe += 1
		else: 
			break
  
# Release all space and windows once done 
	cam.release() 
	cv2.destroyAllWindows() 
	
	######################################################
	x_path = []
	rootdir = 'C:/Users/Kashif/Desktop/Flask Segmentation/data'
	for root, dirs, files in os.walk(rootdir):
		for name in files:
			if name.endswith((".jpg")):
				x_path.append(os.path.join(root, name).replace('\\','/'))

	x_path.sort()  
	print(x_path[0])
	

	for test_img in tqdm(range(len(x_path))):
		image = cv2.imread(x_path[test_img])
		img = cv2.resize(image, (256, 256))
		img = np.float32(img)  / 255 

		image = np.array(img)
		result = model.predict(np.expand_dims(image,axis = 0))
		result = np.argmax(result, axis=3)
		colors = np.array([
		[255, 192 ,203],      
		[255, 160, 122],     
		[255, 105, 180],      
		[205,  92,  92],        
		[255, 165,   0],    
		[255, 255,   0],      
		[165,  42,  42],     
		[0,   0, 255]           
	], dtype=np.int)
		color_image = np.zeros(
			(result.shape[1], result.shape[2], 3), dtype=np.int)
		for i in range(8):
			color_image[result[0] == i] = colors[i]

    #pyplot.figure(figsize=(30, 30))
		fig, (ax0, ax1) = pyplot.subplots(ncols=2,figsize=(15,15))
    
    #ax0.figure(figsize=(10, 10))
		ax0.axis('off')
		ax0.imshow(color_image)
    
    #ax1.figure(figsize=(10, 10))
		ax1.axis('off')
		ax1.imshow(mpimg.imread(x_path[test_img]))
		pyplot.savefig('C:/Users/Kashif/Desktop/Flask Segmentation/Img/'+x_path[test_img].split('/')[-1])
		pyplot.close('all')
		#pyplot.show()
		
		##################################################################
		
	#https://stackoverflow.com/questions/24961127/how-to-create-a-video-from-images-with-ffmpeg
	os.system('ffmpeg -r 10 -i "C:\\Users\\Kashif\\Desktop\\Flask Segmentation\\Img\\frame%01d.jpg"  -vcodec mpeg4 -y movie.mp4')

    
	
	return "Process completed now your video is segmented refer movie.mp4"


	
	
if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080,threaded=False,debug=False)