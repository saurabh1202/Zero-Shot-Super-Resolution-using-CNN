###############################################
  #Assignment-3
  #Zero Shot Super Resolution Using CNN
  #Authored By - Saurabh Nimbalkar & Anish Kale		
###############################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

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
	
		
images = read_image_files()

for image in images:
	im1 = image_to_numpy(image)
	tmp_mainImg = np.pad(im1, 1, padwithzero)
	print(tmp_mainImg.shape)
	exit()






