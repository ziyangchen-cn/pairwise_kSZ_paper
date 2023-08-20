import sys
sys.path.append("/home/chenzy/code/SZ_planck_DESI/")
import os
import warnings
from tqdm import tqdm
import time
from scipy.spatial import KDTree


import numpy as np
import matplotlib.pyplot as plt

def resample_jackknife(the,phi,N_bin,N_bin1):
	N_bin2=N_bin//N_bin1
	#resample
	theta_array=np.percentile(the,np.linspace(0,100,N_bin1+1))
	theta_array[-1]+=0.01
	phi_array=np.zeros((N_bin1,N_bin2+1))
	for ii in range(N_bin1):
		label_tt=np.where((the>=theta_array[ii])&(the<theta_array[ii+1]))
		phi_array[ii,:]=np.percentile(phi[label_tt],np.linspace(0,100,N_bin2+1))
		phi_array[ii,-1]+=0.01
	label_array_temp=np.zeros(N_bin,dtype=np.ndarray)
	for ii in range(N_bin1):
		for jj in range(N_bin2):
			label_array_temp[N_bin2*ii+jj]=np.where((the>=theta_array[ii])&(the<theta_array[ii+1])&(phi>=phi_array[ii,jj])&(phi<phi_array[ii,jj+1]))[0]
	label_array=np.zeros(N_bin,dtype=np.ndarray)
	for i in range(N_bin):
		if i==0:
			label_array[i]=np.hstack(label_array_temp[1:]).flatten();continue
		if i==N_bin-1:
			label_array[i]=np.hstack(label_array_temp[:-1]).flatten();continue

		a1=np.hstack(label_array_temp[:i]).flatten()
		a2=np.hstack(label_array_temp[i+1:]).flatten()

		label_array[i]=np.hstack([a1,a2])
	#print(label_array_temp)
	return label_array, label_array_temp


