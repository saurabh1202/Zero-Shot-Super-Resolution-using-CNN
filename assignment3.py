###############################################
  #Assignment-3
  #Zero Shot Super Resolution Using CNN
  #Authored By - Saurabh Nimbalkar & Anish Kale		
###############################################
import numpy
import cv2
import glob

############## READING IMAGE ###############

def read_image_files():
	images = glob.glob("*.jpg")
	return images


def display_images(images):
	for image in images:
		img = cv2.imread(image)
		print(img,"-----------")
	
images = read_image_files()


#print (images)

display_images(images)






