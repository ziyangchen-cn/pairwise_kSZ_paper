'''
This part is for data preparing.
Included:
1. read NG, ra, dec, zph, lgM from DESI catalog
"gv": group version, "DR8"/"DR9", def="DR8"
"g_posi": "NGC"/"SGC", def="NGC"

2. read CMB map
"cmbtype": def="217", "simca_no_sz" or other frequency

3. sample selection
"N_g": def=60000, expect "condi", the number of group
"range_method":
    "N": the default order in yang group (Ng)
    "M": group mass
    "the": angluar radius
    "condi"; M>M,the>the

4. add AP filter
"ap_the": def=3

return Ng, ra, dec, zph, lgM, T_g


'''

import sys
sys.path.append("/home/chenzy/code/SZ_planck_DESI/")
import os
import warnings
from tqdm import tqdm
import time
from scipy.spatial import KDTree


import numpy as np
import matplotlib.pyplot as plt

from numba import jit

from readdata import *
from funcs import *

#dir
picdir="/home/chenzy/pic/"
datadir="/home/chenzy/code/SZ_planck_DESI/pairwise_ksz/data/"


# ============= AP filter ============
def AP_l(l,the):
#W_ap(l|theta_ap)
	def w_top(x):
		return j1(x)/x
	x=l*the*np.pi/180/60
	wap=4*(w_top(x)-w_top(np.sqrt(2)*x))
	wap[np.where(x==0)]=0
	return wap
def convol_map_AP(cmbmap,nside,the,nest=True):
	t0=time.time()
	l_max=3*nside-1
	if nest:
		cmbmap=hp.reorder(cmbmap,n2r=True)
	aml=hp.sphtfunc.map2alm(cmbmap)
	l_array=hp.sphtfunc.Alm.getlm(l_max,np.arange(len(aml)))
	W_ap=AP_l(l_array[0],the=the)
	aml_ap=aml*W_ap
	cmb_ap=hp.sphtfunc.alm2map(aml_ap,nside)
	cmb_ap=hp.reorder(cmb_ap,r2n=True)
	print("Add AP filter to a sky map, "+str(np.round(time.time()-t0,3))+"s")
	return cmb_ap
def AP_on_gal(the,phi,nside,ap_the,cmap):
#ap_the arcmin
#this one is for theta-space
#not used
	ap_the=ap_the*np.pi/180/60
	if type(ap_the)==int or type(ap_the)==np.float64:
		ap_the=np.full(len(the),ap_the)

	vec=hp.pixelfunc.ang2vec(the,phi)
	T=np.zeros(len(the))
	for i in range(len(vec[:,0])):
		pix_inner=hp.query_disc(nside,vec[i,:],ap_the[i],nest=True)
		pix_outer=hp.query_disc(nside,vec[i,:],ap_the[i]*np.sqrt(2),nest=True)

		a1=np.sum(cmap[pix_inner])
		a2=np.sum(cmap[pix_outer])

		n1=len(pix_inner)
		n2=len(pix_outer)



		T[i]=(a1/n1)-(a2-a1)/(n2-n1)

	return T


def Data_pre_processing(gv="DR8",g_posi="NGC",cmbtype="217",range_method="N",N_g=60000,ap_the=3,the=0,M=0):
	warnings.filterwarnings("ignore")
	print("Data pre-proessing.....")
	print("Group cataloge: ",g_posi,"\nCMB: ",cmbtype,"\nRange: ",range_method,"\nAP filter: ",ap_the," arcmin")
	t0=time.time()
	#read data
	if N_g==60000:
		if range_method=="N":
			Ng,ra,dec,zph,lgM=readDesiGroup_N(g_posi,N=N_g,gv=gv)
		if range_method=="M":
			Ng,ra,dec,zph,lgM=readDesiGroup_M(g_posi,N=N_g,gv=gv)
		if range_method=="the":
			Ng,ra,dec,zph,lgM=readDesiGroup_the(g_posi,N=N_g,gv=gv)
		if range_method=="condi":
			Ng,ra,dec,zph,lgM=readDesiGroup_conditions(g_posi,the,M,gv=gv)
	else:
		if range_method=="N":
			Ng,ra,dec,zph,lgM=readDesiGroup_N(g_posi,N=1000000,gv=gv)
			Ng=Ng[:N_g];ra=ra[:N_g];dec=dec[:N_g];zph=zph[:N_g];lgM=lgM[:N_g]
		if range_method=="M":
			Ng,ra,dec,zph,lgM=readDesiGroup_M(g_posi,N=1000000,gv=gv)
			Ng=Ng[:N_g];ra=ra[:N_g];dec=dec[:N_g];zph=zph[:N_g];lgM=lgM[:N_g]


	if cmbtype=="217" or cmbtype=="smica_no_sz":
		cmbmap,nside=readcmb(cmbtype)
	else:
		cmbmap,cmb_inp,mask,nside=readcmb(cmbtype)

	#ra,dec ->  the, phi (G) -> pixel(G,nest)
	the_gal,phi_gal=radec2thephi(ra,dec)
	group_pix=hp.pixelfunc.ang2pix(nside,the_gal,phi_gal,nest=True)
	#ap_filter
	cmb_ap=convol_map_AP(cmbmap,nside,ap_the)
	T_g=cmb_ap[group_pix]*10**6

	print("Data_pre_processing costs "+str(np.round(time.time()-t0,3))+"s")

	return Ng,ra,dec,zph,lgM,T_g