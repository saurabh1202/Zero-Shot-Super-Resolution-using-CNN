###############################################
  #Assignment-3
  #Zero Shot Super Resolution Using CNN
  #Authored By - Saurabh Nimbalkar & Anish Kale		
###############################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import scipy as sc

############## READING IMAGE ###############

def read_image_files():
	images = glob.glob("*.jpg")
	return images


def image_to_numpy(images):
	img = cv2.imread(image)
	im1 = np.array(img)
		
	"""B = im1.copy()
	G = im1.copy()
	R = im1.copy()
	B[:,:,1] = 0
	B[:,:,2] = 0
	G[:,:,0] = 0
	G[:,:,2] = 0
	R[:,:,0] = 0
	R[:,:,1] = 0
	print (B.shape)
	exit()
	
	print(B,"\n----------")
	print(G,"\n----------")
	print(R,"\n----------")
	print("\n***************\n")
	#return B,G,R
	"""
	return im1
def padwithzero(vector, pad_width, iaxis, kwargs):
	vector[:pad_width[0]] = 0
	vector[-pad_width[1]:] = 0
	return vector

def downsample_using_bicubic_intpln(img):
	down_sampled_list = []
	down_sampled_image = img
	for i in range(1,8):
		down_sampled_image = cv2.resize(down_sampled_image, dsize=(down_sampled_image.shape[1]-1, down_sampled_image.shape[0]-1), interpolation=cv2.INTER_CUBIC)
		down_sampled_list.append(down_sampled_image)
	
	return down_sampled_list
	
images = read_image_files()

for image in images:
	img = image_to_numpy(image)
	#img1 = np.pad(img, 1, padwithzero)
	#print(img.shape)
	down_sampled_list = downsample_using_bicubic_intpln(img)
	#cv2.imshow("original image",img)
	#cv2.waitKey(0)
	#cv2.imshow("downsampled by 7 steps",down_sampled_list[-1])
	#cv2.waitKey(0)
	print(down_sampled_list[-1].shape)
	exit()






