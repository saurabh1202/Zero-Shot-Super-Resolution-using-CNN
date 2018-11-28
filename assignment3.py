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
import math

############## READING IMAGE ###############

def read_image_files():
	images = glob.glob("*.jpg")	
	return images


def image_to_numpy(image):
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

def create_target_image_set(img):
	down_sampled_list = []
	down_sampled_list.append(img)
	for i in range(7):
		down_sampled_image = cv2.resize(img, dsize=(img.shape[1]-i-1, img.shape[0]-i-1), interpolation=cv2.INTER_CUBIC)
		down_sampled_list.append(down_sampled_image)
	
	
	return down_sampled_list

def create_train_image_set(target_set):
	up_sampled_list = []
	for i in range(7):
		up_sampled_image = cv2.resize(target_set[i+1], dsize=(target_set[i+1].shape[1]+1, target_set[i+1].shape[0]+1), interpolation=cv2.INTER_CUBIC)
		up_sampled_list.append(up_sampled_image)
	
	return up_sampled_list
	
class Convolution:
	
	def __init__(self,shape,in_channels,out_channels,kernal_size=3,stride=1):
		self.shape = shape
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernal_size = kernal_size
		self.stride = 1
		self.batch_size = 1
		volume = shape[0] * shape[1] * shape[2]
		#print(volume)
		#exit()
		std_dev = math.sqrt(2/volume)
		self.weights = np.random.normal(0, std_dev, (self.kernal_size, self.kernal_size, in_channels,out_channels))
		#print(self.weights)
		#exit()               
		self.bias = np.random.normal(0, std_dev, self.out_channels)
		self.weight_gradients = np.zeros(self.weight.shape)
		self.bias_gradients = np.zeros(self.bias.shape)
	
	def forward(self,inp):
		print (inp)
		exit()
		'''self.in_shape = inp.shape
		self.col_weights = self.weights.reshape([-1,self.out_channels])
		#print(col_weights)
		#exit()
		self.eta = np.zeros((self.in_shape[0],self.in_shape[1] // self.stride, self.in_shape[2] // self.stride,self.out_channels)
		#inp = np.pad(inp, ((0, 0), (self.kernal_size // 2, self.kernal_size // 2), (self.kernal_size // 2, self.kernal_size // 2), (0, 0)),'constant', constant_values=0)
		#print("executing")
		#exit()
		self.out_shape = self.eta.shape
		self.col_image = []
		convolution_output = np.zeros(self.out_shape)
		#print (input)
		#exit()
		for i in range(self.batch_size):
		img = inp[i][np.newaxis,:]
			self.col_image_i = im2col(img, self.kernal_size, self.stride)
			convolution_output[i]=np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
			self.col_image.append(self.col_image_i)
		self.col_image = np.array(self.col_image)
		return convolution_output
'''
class MSE:
	def _init_(self):
		print("Mean square error")
	
	def loss_calculation(self,output,target):
		shape = output.shape
		N = 1
		for i in shape:
			N = N * i
		loss = np.sum(np.square(target-output))/N
		return loss
	def gradient(self,output,target):
		Dx = -(target-output)
		return Dx
		
class Relu:
	def _init_(self):
		print("Rectified Linear Unit")
	
	def forward(self,output):
		self.output = output
		if output > 0:
			return output
		else:
			return 0
	def backward(self):
		pass
		



		
def im2col(img,kernal_size,stride):
	image_col = []
	for i in range(0, img.shape[1] - kernal_size + 1, stride):
		for j in range(0, img.shape[2] - kernal_size + 1, stride):
			col = img[:, i:i + kernal_size, j:j + kernal_size, :].reshape([-1])
			image_col.append(col)
	image_col = np.array(image_col)
	return image_col
	
		
		


def build_and_train_network(train_set_1,target_set_1,input_blurred_image,no_of_epochs,learning_rate):
	in_shape = train_set_1[0].shape
	conv_obj_1 = Convolution(in_shape,3,64)
	conv_obj_1 = Convolution(in_shape,64,64)
	conv_obj_1 = Convolution(in_shape,64,64)
	conv_obj_1 = Convolution(in_shape,64,64)
	conv_obj_1 = Convolution(in_shape,64,64)
	conv_obj_1 = Convolution(in_shape,64,64)
	conv_obj_1 = Convolution(in_shape,64,64)
	conv_obj_1 = Convolution(in_shape,64,64)
	output = Convolution(shape,64,3)
	relu = Relu()
	mse = MSE()
	list_of_losses = []
	
	
	
	for i in no_of_epochs:
		for j in range(len(train_set_1)):
			conv_1 = Convolution(in_shape,3,64)
			conv_1_out = conv_1.forward(train_set_1[j])
			#relu_1 = Relu.forward
	



	
img_files = read_image_files()
images = []
for img in img_files:
	im = image_to_numpy(img)
	images.append(im)	
	
image = images[1]
#print(image.shape)
#exit()
target = create_target_image_set(image)
train_set = create_train_image_set(target)
target_set = target[0:len(target)-1]
train_set_1 = []
target_set_1 = []
for i in train_set:
	i = i[np.newaxis,:,:,:]
	train_set_1.append(i)

for i in target_set:
	i = i[np.newaxis,:,:,:]
	target_set_1.append(i)
	
#for i in train_set_1:
	#print(i.shape)
#for i in target_set_1:
	#print(i.shape)
#exit()


input_blurred_image = cv2.resize(target_set[0],dsize=(target_set[0].shape[1]+1,target_set[0].shape[0]+1), interpolation=cv2.INTER_CUBIC)
#print(input_blurred_image.shape)

#exit()
input_blurred_image = input_blurred_image[np.newaxis,:,:,:]
print(input_blurred_image.shape)
exit()
no_of_epochs = 500
learning_rate = 0.00001
output_sharp_image = build_and_train_network(train_set_1,target_set_1,input_blurred_image,no_of_epochs,learning_rate)














