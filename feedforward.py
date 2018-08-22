import os
import struct 
import numpy as np

def load_mnist(path,kind='train',kind1='t10k'):
	labels_path = os.path.join(path,'%s-labels-idx1-ubyte'%kind)
	images_path = os.path.join(path,'%s-images-idx3-ubyte'%kind)
	labels_path1 = os.path.join(path,'%s-labels-idx1-ubyte'%kind1)
	images_path1 = os.path.join(path,'%s-images-idx3-ubyte'%kind1)
	
	with open(labels_path,'rb') as lbpath:
		magic,n = struct.unpack('>II',
			lbpath.read(8))
		labels = np.fromfile(lbpath,dtype=np.uint8)

	with open(images_path,'rb') as imgpath:
		magic,num,rows,cols =struct.unpack(">IIII",
			imgpath.read(16))
		images = np.fromfile(imgpath,
			dtype=np.uint8).reshape(len(labels),784)
		images = ((images/255.)-.5)*2
	
	with open(labels_path1,'rb') as lbpath1:
		magic,n = struct.unpack('>II',
			lbpath1.read(8))
		labels1 = np.fromfile(lbpath1,dtype=np.uint8)
	with open(images_path1,'rb') as imgpath1:
		magic,num,rows,cols =struct.unpack(">IIII",
			imgpath1.read(16))
		images1 = np.fromfile(imgpath1,
			dtype=np.uint8).reshape(len(labels1),784)
		images1 = ((images1/255.)-.5)*2

	return images,labels,images1,labels1

X_train, y_train,X_test,y_test = load_mnist('', kind='train',kind1='t10k')
print('Rows: %d, columns: %d'
	% (X_train.shape[0], X_train.shape[1]))
print('Rows: %d, columns: %d'
	% (X_test.shape[0], X_test.shape[1]))

np.savez_compressed('mnist_scaled.npz',
	X_train=X_train,
	y_train=y_train,
	X_test=X_test,
	y_test=y_test)

mnist = np.load('mnist_scaled.npz')
X_test = mnist['X_test']
y_test = mnist['y_test']
print(y_test.shape[0])
print(X_test.shape)
