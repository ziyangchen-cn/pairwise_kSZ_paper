import sys
sys.path.append("/home/chenzy/code/SZ_planck_DESI/")
import os
import warnings
from tqdm import tqdm
import time
from scipy.spatial import KDTree


import numpy as np


def chi_square(v1,v2,C_):
	v1=np.mat(v1)
	v2=np.mat(v2)
	return np.dot(v1,np.dot(C_,v2.T))[0,0]
def fit_para(s_t,s_m,C_):
	mean=chi_square(s_m,s_t,C_)/chi_square(s_t,s_t,C_)
	sigma=np.sqrt(1/chi_square(s_t,s_t,C_))

	return mean,sigma
	
def pseudo_inverse(C,n):
	N=C.shape[0]
	w,v =np.linalg.eig(C)
	w=np.sort(w)

	if n==0:
		return np.linalg.pinv(C)
	else:
		return np.linalg.pinv(C,(w[n]+w[n-1])/2/w[-1])
def chi_square(v1,v2,C_):
	v1=np.mat(v1)
	v2=np.mat(v2)
	return np.dot(v1,np.dot(C_,v2.T))[0,0]

def error_esti_jackknife(N_bin,samples):
#samples.shape=(r_bin,N_bin)
	r_bin=samples.shape[0]
	N_bin=samples.shape[1]
	s_mean=np.mean(samples,axis=1)
	corv=np.zeros((r_bin,r_bin))
	for ii in range(r_bin):
		for jj in range(r_bin):
			for kk in range(N_bin):
				corv[ii,jj]+=(samples[ii,kk]-s_mean[ii])*(samples[jj,kk]-s_mean[jj])
	corv*=(N_bin-1.)/N_bin
	#esti=(N_bin-r_bin-1.)/(N_bin-1)*np.linalg.inv(corv)
	#print("esti:\n",esti)
	return s_mean,corv


def signal_and_noise(jk_sample,theory_template,dps=2,n_ev=2):
	
	r_bin=jk_sample.shape[0]-dps
	N_bin=jk_sample.shape[1]
	s_mean,s_corv=error_esti_jackknife(N_bin,jk_sample)
	C_=(N_bin-r_bin-2.0)/(N_bin-1.)*pseudo_inverse(s_corv[dps:,dps:],n_ev)
	coef_fit,coef_sigma=fit_para(theory_template[dps:],s_mean[dps:],C_)
	chi2_null=chi_square(s_mean[dps:],s_mean[dps:],C_)
	chi2_model=chi_square((s_mean-coef_fit*theory_template)[dps:],(s_mean-coef_fit*theory_template)[dps:],C_)

	return s_mean,s_corv,coef_fit,coef_sigma,chi2_null,chi2_model