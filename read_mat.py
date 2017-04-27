import scipy.io as sio  
import h5py

def read_mat(url):
	data = sio.loadmat(url)  
	return data

def read_mat_v(url,name):
	with h5py.File(url, 'r') as f:
		data = f[name][()]
	return data
