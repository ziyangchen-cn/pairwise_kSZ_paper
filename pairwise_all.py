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

#======================== AP filter =============================
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


#======================== pairwise =============================
def pairwise_old(deco,rao,zo,To,lgMo,alpha,dd):
	def mean_T_corr(T,z,sigma_z):
		z_bin=np.linspace(0.1,1,400)
		z_=(z_bin[1:]+z_bin[:-1])/2
		T_mean_ap=np.zeros(len(z_))

		for j in range(len(z_bin)-1):
			weight_z=np.exp(-(z_[j]-z)**2/2/sigma_z**2)
			T_mean_ap[j]=np.sum(T*weight_z)/np.sum(weight_z)
		f = interpolate.interp1d(z_, T_mean_ap,bounds_error=False, fill_value=0)
		T_corr=f(z)
		return T-T_corr
	@jit(nopython=True)
	def pairwise_for(p,sum_c,s_M,N,x,y,z,cosDec,cosRa,sinDec,sinRa,T,dc,dd):
		for i in (range(len(T))):
			#if i%1000==0:print(i)
			for j in range(i+1,len(T)):
				r=np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i]-z[j])**2)
				if r>300:
					continue
				label_r=int(r/dd)
				cos=cosDec[i]*cosDec[j]*cosRa[i]*cosRa[j]+cosDec[i]*cosDec[j]*sinRa[i]*sinRa[j]+sinDec[i]*sinDec[j]
				c=(M[i]+M[j])*(dc[i]-dc[j])*(1+cos)/2/np.sqrt(dc[i]**2+dc[j]**2-2*dc[i]*dc[j]*cos)
				p[label_r]-=(T[i]-T[j])*c
				sum_c[label_r]+=c**2
				N[label_r]+=1
				s_M[label_r]+=(M[i]+M[j])


	dec=deco*np.pi/180
	ra=rao*np.pi/180
	dc=z2dc(zo)
	cosDec=np.cos(dec)
	sinDec=np.sin(dec)
	cosRa=np.cos(ra)
	sinRa=np.sin(ra)
	x=dc*cosDec*cosRa
	y=dc*cosDec*sinRa
	z=dc*sinDec
	T=mean_T_corr(To,zo,0.05)
	M=10**(lgMo*alpha)/np.mean(10**(lgMo*alpha))/2


	p=np.zeros(300//dd)
	sum_c=np.zeros(300//dd)
	s_M=np.zeros(300//dd)
	N=np.zeros(300//dd)

	pairwise_for(p,sum_c,s_M,N,x,y,z,cosDec,cosRa,sinDec,sinRa,T,dc,dd)

	p=p/sum_c*(s_M/N)
	#print(list(p),sum_c,N)
	return p
def pairwise_2(deco,rao,zo,To,lgMo,alpha,dd,sigma2=0):
	@jit(nopython=True)
	def pairwise_for(s_TC,s_C2,s_C,s_T,s_M,N,x,y,z,cosDec,cosRa,sinDec,sinRa,T,M,dc,dd):
		for i in (range(len(T))):
			#if i%1000==0:print(i)
			for j in range(i+1,len(T)):
				r=np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i]-z[j])**2)
				if r>300:
					continue
				label_r=int(r/dd)
				cos=cosDec[i]*cosDec[j]*cosRa[i]*cosRa[j]+cosDec[i]*cosDec[j]*sinRa[i]*sinRa[j]+sinDec[i]*sinDec[j]
				c=(M[i]+M[j])*(dc[i]-dc[j])*(1+cos)/2/np.sqrt(dc[i]**2+dc[j]**2-2*dc[i]*dc[j]*cos)
				#p[label_r]-=(T[i]-T[j])*c
				#sum_c[label_r]+=c**2
				s_T[label_r]+=(T[i]-T[j])
				s_C[label_r]+=c
				s_C2[label_r]+=c**2
				s_TC[label_r]+=(T[i]-T[j])*c
				s_M[label_r]+=(M[i]+M[j])
				N[label_r]+=1


	dec=deco*np.pi/180
	ra=rao*np.pi/180
	dc=z2dc(zo)
	cosDec=np.cos(dec)
	sinDec=np.sin(dec)
	cosRa=np.cos(ra)
	sinRa=np.sin(ra)
	x=dc*cosDec*cosRa
	y=dc*cosDec*sinRa
	z=dc*sinDec


	#p=np.zeros(300//dd)
	#sum_c=np.zeros(300//dd)
	s_TC=np.zeros(300//dd)
	s_C2=np.zeros(300//dd)
	s_C=np.zeros(300//dd)
	s_T=np.zeros(300//dd)
	s_M=np.zeros(300//dd)
	N=np.zeros(300//dd)
	T=To
	M=10**(lgMo*alpha)/np.mean(10**(lgMo*alpha))/2

	pairwise_for(s_TC,s_C2,s_C,s_T,s_M,N,x,y,z,cosDec,cosRa,sinDec,sinRa,T,M,dc,dd)

	#p/=sum_c
	#print(list(p),sum_c,N)
	T_pksz=(s_TC/N-s_T/N*s_C/N)/(s_C2/N-(s_C/N)**2)*(s_M/N)
	n0=(s_T-T_pksz*s_C)/N

	if sigma2:

		M_fisher=np.zeros((2,2,len(N)))
		M_vari=np.zeros((2,2,len(N)))
		M_fisher[0,0]=s_C2/sigma2
		M_fisher[1,1]=N/sigma2
		M_fisher[0,1]=s_C/sigma2
		M_fisher[1,0]=s_C/sigma2
		for i in range(len(N)):
			M_vari[:,:,i]=np.linalg.inv(M_fisher[:,:,i])
		return T_pksz, n0, M_vari
	else:
		return T_pksz, n0
def pairwise_3a(deco,rao,zo,To,lgMo,alpha,dd,sigma2=0):
	@jit(nopython=True)
	def pairwise_for(s_TZ,s_CZ,s_TC,s_C2,s_Z2,s_M,N,x,y,z,cosDec,cosRa,sinDec,sinRa,T,M,dc,dd):
		for i in (range(len(T))):
			#if i%1000==0:print(i)
			for j in range(i+1,len(T)):
				r=np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i]-z[j])**2)
				if r>300:
					continue
				label_r=int(r/dd)
				cos=cosDec[i]*cosDec[j]*cosRa[i]*cosRa[j]+cosDec[i]*cosDec[j]*sinRa[i]*sinRa[j]+sinDec[i]*sinDec[j]
				c=(M[i]+M[j])*(dc[i]-dc[j])*(1+cos)/2/np.sqrt(dc[i]**2+dc[j]**2-2*dc[i]*dc[j]*cos)
				zij=z[i]-z[j]
				#p[label_r]-=(T[i]-T[j])*c
				#sum_c[label_r]+=c**2
				s_CZ[label_r]+=c*zij
				s_C2[label_r]+=c**2
				s_Z2[label_r]+=zij**2
				s_TC[label_r]+=(T[i]-T[j])*c
				s_TZ[label_r]+=(T[i]-T[j])*zij
				s_M[label_r]+=(M[i]+M[j])
				N[label_r]+=1


	dec=deco*np.pi/180
	ra=rao*np.pi/180
	dc=z2dc(zo)
	cosDec=np.cos(dec)
	sinDec=np.sin(dec)
	cosRa=np.cos(ra)
	sinRa=np.sin(ra)
	x=dc*cosDec*cosRa
	y=dc*cosDec*sinRa
	z=dc*sinDec


	#p=np.zeros(300//dd)
	#sum_c=np.zeros(300//dd)
	s_TZ=np.zeros(300//dd)
	s_CZ=np.zeros(300//dd)
	s_TC=np.zeros(300//dd)
	s_C2=np.zeros(300//dd)
	s_Z2=np.zeros(300//dd)
	s_M=np.zeros(300//dd)
	N=np.zeros(300//dd)
	T=To
	M=10**(lgMo*alpha)/np.mean(10**(lgMo*alpha))/2

	pairwise_for(s_TZ,s_CZ,s_TC,s_C2,s_Z2,s_M,N,x,y,z,cosDec,cosRa,sinDec,sinRa,T,M,dc,dd)

	#p/=sum_c
	#print(list(p),sum_c,N)
	T_pksz=((s_TC*s_Z2-s_TZ*s_CZ)/(s_C2*s_Z2-s_CZ**2))*(s_M/N)
	n1=(s_C2*s_TZ-s_TC*s_CZ)/(s_C2*s_Z2-s_CZ**2)

	if sigma2:
		M_fisher=np.zeros((2,2,len(N)))
		M_vari=np.zeros((2,2,len(N)))
		M_fisher[0,0,:]=s_C2/sigma2
		M_fisher[1,1,:]=s_Z2/sigma2
		M_fisher[0,1,:]=s_CZ/sigma2
		M_fisher[1,0,:]=s_CZ/sigma2
		for i in range(len(N)):
			M_vari[:,:,i]=np.linalg.inv(M_fisher[:,:,i])
		return T_pksz,n1,M_vari
	else:
		return T_pksz,n1
def pairwise_3b(deco,rao,zo,To,lgMo,alpha,dd,sigma2=0):
	@jit(nopython=True)
	def pairwise_for(s_TC,s_TZ,s_T,s_C2,s_Z2,s_CZ,s_C,s_Z,s_M,N,x,y,z, redshift, cosDec,cosRa,sinDec,sinRa,T,M,dc,dd):
		for i in (range(len(T))):
			#if i%1000==0:print(i)
			for j in range(i+1,len(T)):
				r=np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i]-z[j])**2)
				if r>300:
					continue
				label_r=int(r/dd)
				cos=cosDec[i]*cosDec[j]*cosRa[i]*cosRa[j]+cosDec[i]*cosDec[j]*sinRa[i]*sinRa[j]+sinDec[i]*sinDec[j]
				c=(M[i]+M[j])*(dc[i]-dc[j])*(1+cos)/2/np.sqrt(dc[i]**2+dc[j]**2-2*dc[i]*dc[j]*cos)
				s_M[label_r]+=(M[i]+M[j])
				zij = (redshift[i]-redshift[j])
				#p[label_r]-=(T[i]-T[j])*c
				#sum_c[label_r]+=c**2
				s_CZ[label_r]+=c*zij
				s_C2[label_r]+=c**2
				s_Z2[label_r]+=zij**2
				s_C[label_r]+=c
				s_Z[label_r]+=zij
				s_TC[label_r]+=(T[i]-T[j])*c
				s_TZ[label_r]+=(T[i]-T[j])*zij
				s_T[label_r]+=(T[i]-T[j])
				N[label_r]+=1

	print("THIS IS NEW")
	dec=deco*np.pi/180
	ra=rao*np.pi/180
	dc=z2dc(zo)
	cosDec=np.cos(dec)
	sinDec=np.sin(dec)
	cosRa=np.cos(ra)
	sinRa=np.sin(ra)
	x=dc*cosDec*cosRa
	y=dc*cosDec*sinRa
	z=dc*sinDec


	s_TC=np.zeros(300//dd)
	s_TZ=np.zeros(300//dd)
	s_T=np.zeros(300//dd)
	s_C2=np.zeros(300//dd)
	s_Z2=np.zeros(300//dd)
	s_CZ=np.zeros(300//dd)
	s_C=np.zeros(300//dd)
	s_Z=np.zeros(300//dd)
	s_M=np.zeros(300//dd)
	N=np.zeros(300//dd)
	T=To
	M=10**(lgMo*alpha)/np.mean(10**(lgMo*alpha))/2

	pairwise_for(s_TC,s_TZ,s_T,s_C2,s_Z2,s_CZ,s_C,s_Z,s_M,N,x,y,z,zo,cosDec,cosRa,sinDec,sinRa,T,M,dc,dd)

	Matrix_C=np.zeros((3,3,len(N)))
	Matrix_C[0,0,:]=s_C2/N
	Matrix_C[1,1,:]=s_Z2/N
	Matrix_C[2,2,:]=1
	Matrix_C[0,1,:]=s_CZ/N
	Matrix_C[0,2,:]=s_C/N
	Matrix_C[1,2,:]=s_Z/N
	Matrix_C[1,0,:]=s_CZ/N
	Matrix_C[2,0,:]=s_C/N
	Matrix_C[2,1,:]=s_Z/N
	b=np.zeros((3,len(N)))
	b[0,:]=s_TC/N
	b[1,:]=s_TZ/N
	b[2,:]=s_T/N

	#solve linear equation
	t0=time.time()
	x=np.zeros((3,len(N)))
	for i in range(len(N)):
		x[:,i] = np.linalg.solve(Matrix_C[:,:,i],b[:,i])
	x[0,:]=x[0,:]*(s_M/N)

	if sigma2:
		print("calculate fisher matrix")
		M_fisher=np.zeros((3,3,len(N)))
		M_vari=np.zeros((3,3,len(N)))
		M_fisher[0,0,:]=s_C2/sigma2
		M_fisher[1,1,:]=s_Z2/sigma2
		M_fisher[2,2,:]=N/sigma2
		M_fisher[0,1,:]=s_CZ/sigma2
		M_fisher[0,2,:]=s_C/sigma2
		M_fisher[1,2,:]=s_Z/sigma2
		M_fisher[1,0,:]=s_CZ/sigma2
		M_fisher[2,0,:]=s_C/sigma2
		M_fisher[2,1,:]=s_Z/sigma2
		for i in range(len(N)):
			M_vari[:,:,i]=np.linalg.inv(M_fisher[:,:,i])
		return x,M_vari
	else:
		return x, s_Z2/N
def pairwise_3c(deco,rao,zo,To,lgMo,alpha,dd,sigma2=0):
	@jit(nopython=True)
	def pairwise_for(s_TC,s_TZ,s_T,s_C2,s_Z2,s_CZ,s_C,s_Z,s_M,N,x,y,z, redshift, cosDec,cosRa,sinDec,sinRa,T,M,dc,dd):
		for i in (range(len(T))):
			#if i%1000==0:print(i)
			for j in range(i+1,len(T)):
				r=np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i]-z[j])**2)
				if r>300:
					continue
				label_r=int(r/dd)
				cos=cosDec[i]*cosDec[j]*cosRa[i]*cosRa[j]+cosDec[i]*cosDec[j]*sinRa[i]*sinRa[j]+sinDec[i]*sinDec[j]
				c=(M[i]+M[j])*(dc[i]-dc[j])*(1+cos)/2/np.sqrt(dc[i]**2+dc[j]**2-2*dc[i]*dc[j]*cos)
				s_M[label_r]+=(M[i]+M[j])
				zij = (redshift[i]-redshift[j])**2
				#p[label_r]-=(T[i]-T[j])*c
				#sum_c[label_r]+=c**2
				s_CZ[label_r]+=c*zij
				s_C2[label_r]+=c**2
				s_Z2[label_r]+=zij**2
				s_C[label_r]+=c
				s_Z[label_r]+=zij
				s_TC[label_r]+=(T[i]-T[j])*c
				s_TZ[label_r]+=(T[i]-T[j])*zij
				s_T[label_r]+=(T[i]-T[j])
				N[label_r]+=1

	print("THIS IS NEW")
	dec=deco*np.pi/180
	ra=rao*np.pi/180
	dc=z2dc(zo)
	cosDec=np.cos(dec)
	sinDec=np.sin(dec)
	cosRa=np.cos(ra)
	sinRa=np.sin(ra)
	x=dc*cosDec*cosRa
	y=dc*cosDec*sinRa
	z=dc*sinDec


	s_TC=np.zeros(300//dd)
	s_TZ=np.zeros(300//dd)
	s_T=np.zeros(300//dd)
	s_C2=np.zeros(300//dd)
	s_Z2=np.zeros(300//dd)
	s_CZ=np.zeros(300//dd)
	s_C=np.zeros(300//dd)
	s_Z=np.zeros(300//dd)
	s_M=np.zeros(300//dd)
	N=np.zeros(300//dd)
	T=To
	M=10**(lgMo*alpha)/np.mean(10**(lgMo*alpha))/2

	pairwise_for(s_TC,s_TZ,s_T,s_C2,s_Z2,s_CZ,s_C,s_Z,s_M,N,x,y,z,zo,cosDec,cosRa,sinDec,sinRa,T,M,dc,dd)

	Matrix_C=np.zeros((3,3,len(N)))
	Matrix_C[0,0,:]=s_C2/N
	Matrix_C[1,1,:]=s_Z2/N
	Matrix_C[2,2,:]=1
	Matrix_C[0,1,:]=s_CZ/N
	Matrix_C[0,2,:]=s_C/N
	Matrix_C[1,2,:]=s_Z/N
	Matrix_C[1,0,:]=s_CZ/N
	Matrix_C[2,0,:]=s_C/N
	Matrix_C[2,1,:]=s_Z/N
	b=np.zeros((3,len(N)))
	b[0,:]=s_TC/N
	b[1,:]=s_TZ/N
	b[2,:]=s_T/N

	#solve linear equation
	t0=time.time()
	x=np.zeros((3,len(N)))
	for i in range(len(N)):
		x[:,i] = np.linalg.solve(Matrix_C[:,:,i],b[:,i])
	x[0,:]=x[0,:]*(s_M/N)

	if sigma2:
		print("calculate fisher matrix")
		M_fisher=np.zeros((3,3,len(N)))
		M_vari=np.zeros((3,3,len(N)))
		M_fisher[0,0,:]=s_C2/sigma2
		M_fisher[1,1,:]=s_Z2/sigma2
		M_fisher[2,2,:]=N/sigma2
		M_fisher[0,1,:]=s_CZ/sigma2
		M_fisher[0,2,:]=s_C/sigma2
		M_fisher[1,2,:]=s_Z/sigma2
		M_fisher[1,0,:]=s_CZ/sigma2
		M_fisher[2,0,:]=s_C/sigma2
		M_fisher[2,1,:]=s_Z/sigma2
		for i in range(len(N)):
			M_vari[:,:,i]=np.linalg.inv(M_fisher[:,:,i])
		return x,M_vari
	else:
		return x, s_Z2/N

def cal_pair_mean(pair_r_len, pair_A, r_max, dd):
        r_bin = np.arange(0, r_max+1, dd)
        N = np.histogram(pair_r_len, bins=r_bin, )[0]
        mean = np.histogram(pair_r_len, bins=r_bin, weights=pair_A)[0]
        mean = mean/N

        return mean

def cal_pairwise_corr(deco,rao,redshift,To,lgMo,alpha,dd,sigma2=0, r_max=300):
        print("corr", len(deco))
        dec=deco*np.pi/180;ra=rao*np.pi/180
        dc=z2dc(redshift)
        cosDec=np.cos(dec);sinDec=np.sin(dec)
        cosRa=np.cos(ra);sinRa=np.sin(ra)
        GroupPos = np.zeros((len(deco), 3))
        GroupPos[:,0]=dc*cosDec*cosRa
        GroupPos[:,1]=dc*cosDec*sinRa
        GroupPos[:,2]=dc*sinDec
        M=10**(lgMo*alpha)/np.mean(10**(lgMo*alpha))/2

        print(dc)
        group_tree=KDTree(GroupPos[:20000,:])
        print(1)
        pair_index=np.array(list(group_tree.query_pairs(r=r_max)))
        i = pair_index[:,0]
        j = pair_index[:,1]
        print("Number of pairs", pair_index.shape)

        pair_r_len = np.sqrt(np.sum((GroupPos[pair_index[:,1], :]-GroupPos[pair_index[:,0], :])**2, axis=1))

        cos=cosDec[i]*cosDec[j]*cosRa[i]*cosRa[j]+cosDec[i]*cosDec[j]*sinRa[i]*sinRa[j]+sinDec[i]*sinDec[j]
        pair_C = (M[i]+M[j])**alpha*(dc[i]-dc[j])*(1+cos)/2/np.sqrt(dc[i]**2+dc[j]**2-2*dc[i]*dc[j]*cos)
        pair_Z = redshift[i]-redshift[j]
        #weight_M = (M[i]+M[j])/2
        pair_T = (To[i]-To[j])

        # n0
        Mat = np.zeros((2, 2, 300//dd))
        b = np.zeros((2, 300//dd))
        Mat[0,0,:] = cal_pair_mean(pair_r_len, pair_C**2, r_max, dd)
        Mat[0,1,:] = cal_pair_mean(pair_r_len, pair_C, r_max, dd)
        Mat[1,1,:] = np.ones(300//dd)
        Mat[1,0,:] = Mat[0,1,:]
        b[0,:] = cal_pair_mean(pair_r_len, pair_T*pair_C, r_max, dd)
        b[1,:] = cal_pair_mean(pair_r_len, pair_T, r_max, dd)

        x = np.zeros((2, 300//dd))
        for i in range(300//dd):
                x[:,i] = np.linalg.solve(Mat[:,:,i], b[:,i])

        return x, s_Z2/N
 



#======================== jackknife  =============================
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


#========================= template ==============================

def fit_para(s_t,s_m,C_):
	mean=chi_square(s_m,s_t,C_)/chi_square(s_t,s_t,C_)
	sigma=np.sqrt(1/chi_square(s_t,s_t,C_))
	'''
	tou_r=np.linspace(tou_mean-4*tou_sigma,tou_mean+4*tou_sigma,50)
	chi2_r=np.zeros(len(tou_r))
	for i in range(len(tou_r)):
		chi2_r[i]=chi_square((s_m-tou_r[i]*s_t),C_)
	return tou_mean,tou_sigma,tou_r,chi2_r
	'''
	return mean,sigma


def theory_pairwise(z=0.4,b=3.68,sigmaz=0.01,figplot=0):
	minkh=-5
	maxkh=2
	npoints=100000
	dlnk=(maxkh-minkh)*1./npoints
	print(dlnk)
	k=10**np.linspace(minkh,maxkh,npoints)
	dlnk=np.mean(np.log(k[1:])-np.log(k[:-1]))
	print(dlnk)
	r_min=-1
	r_max=np.log10(300)
	r_npoints=10000
	r=10**np.linspace(r_min,r_max,r_npoints) #Mpc/h
	dlnr=(r_max-r_min)*1./r_npoints
	print(dlnr)
	dlnr=np.mean(np.log(r[1:])-np.log(r[:-1]))
	print(dlnr)



	#p(k) from camb
	t0=time.time()
	pars = camb.CAMBparams()
	pars.set_cosmology(H0=100*h, ombh2=ombh2, omch2=omch2)
	pars.InitPower.set_params(As=2.105e-9,ns=0.9665,r=0)
	pars.NonLinear = model.NonLinear_both
	pars.set_matter_power(redshifts=[z], kmax=1100.0)
	#pars.NonLinear = model.NonLinear_both
	results = camb.get_results(pars)
	#kden, z_nonlin, psden= results.get_linear_matter_power_spectrum(hubble_units=True, k_hunit=True)# kden [h/Mpc]
	kden, z_nonlin, psden= results.get_matter_power_spectrum(minkh=10**minkh, maxkh=10**maxkh, npoints=npoints)# kden [h/Mpc]
	print("camb power spectrum ",np.round(time.time()-t0,3)," s")

	psden=psden[0]

	#1
	t0=time.time()
	kr=r.reshape(-1,1)*kden.reshape(1,-1)
	j0=np.sin(kr)/kr
	del kr
	cor_1=np.sum((kden**3*psden).reshape(1,-1)*j0,axis=1)/2/np.pi**2*dlnk #as a func of r Mpc/h
	print("camb power spectrum ",np.round(time.time()-t0,3)," s")
	#plt.plot(kden,(kden**3*psden*j0[10,:]),"b")
	#plt.plot(r,cor_1,"r")
	#plt.yscale("log")
	#plt.savefig(picdir+"cor1.png");exit()

	#2
	cor_2=np.zeros(len(r))
	for i in range(len(r)):
		cor_2[i]=np.sum(cor_1[:i+1]*r[:i+1]**3)/np.sum(r[:i+1]**3)
		#cor_2[i]=3*np.sum(cor_1[:i+1]*r[:i+1]**3*dlnr)/r[i]**3

	#coef
	cosmo = FlatLambdaCDM(H0=100*h, Om0=(omch2+ombh2)/h**2, Tcmb0=2.725)
	H=cosmo.H(z).value
	E=Omega_m*(1+z)**3+(1-Omega_m)
	f=(Omega_m*(1+z)**3/E)**0.55
	coef=-2.0/3*H/(1+z)*f

	T_pksz=coef*((r/h*b*cor_2)/(1+cor_1*b**2))
	print(T_pksz)

	#photo-z correct
	sigmad=c_speed*sigmaz/H

	T_pksz_ph1=T_pksz*(1-np.exp(-(r/h)**2/2/sigmad**2))
	T_pksz_ph2=T_pksz/np.sqrt(1+(np.sqrt(2)*sigmad/(r/h)/np.sqrt(2/np.pi))**2)


	if figplot:
		plt.figure()
		plt.plot(r,T_pksz,label="$V_{12}(r)$")
		plt.plot(r,T_pksz_ph2,label="$V_{12}(r)$ ph-z")
		plt.xlim(0,300)
		plt.ylim(-200,0)
		plt.legend()
		plt.xlabel("r [Mpc]")
		plt.ylabel("km/s")
		plt.savefig(picdir+"pairwise_theory.png",dpi=300)

	print(r)
	f1 = interpolate.interp1d(r, T_pksz_ph1)
	f2 = interpolate.interp1d(r, T_pksz_ph2)
	f = interpolate.interp1d(r, T_pksz)

	return f,f1,f2

def theory_pairwise_halomod(z=0.3,b=3.68,sigmaz=0.01,figplot=0):
	import halomod

	model = halomod.TracerHaloModel(z=z,transfer_model='EH',rnum=1000,rmin=0.1,rmax=300,hod_model='Zehavi05',hod_params={"M_min": 12,"M_1": 12.8,'alpha': 1.05,'central': True},dr_table=0.01,dlnk=0.01,dlog10m=0.05)


	r=model.r	#mpc/h
	cor_1=model.corr_auto_matter

	#2
	cor_2=np.zeros(len(r))
	for i in range(len(r)):
		cor_2[i]=np.sum(cor_1[:i+1]*r[:i+1]**3)/np.sum(r[:i+1]**3)
		#cor_2[i]=3*3*np.sum(cor_1[:i+1]*r[:i+1]**3*dlnr)/r[i]**3

	#coef
	cosmo = FlatLambdaCDM(H0=100*h, Om0=Omega_m, Tcmb0=2.725)
	H=cosmo.H(z).value
	E=Omega_m*(1+z)**3+(1-Omega_m)
	f=(Omega_m*(1+z)**3/E)**0.55
	coef=-2.0/3*H/(1+z)*f

	T_pksz=coef*(((r)*b*cor_2)/(1+cor_1*b**2))

	#photo-z correct
	sigmad=c_speed*sigmaz/H

	T_pksz_ph1=T_pksz*(1-np.exp(-r**2/2/sigmad**2))
	T_pksz_ph2=T_pksz/np.sqrt(1+(np.sqrt(2)*sigmad/r/np.sqrt(2/np.pi))**2)


	if figplot:
		plt.figure()
		plt.plot(r,T_pksz,label="$V_{12}(r)$")
		plt.plot(r,T_pksz_ph2,label="$V_{12}(r)$ ph-z")
		plt.xlim(0,300)
		plt.ylim(-200,0)
		plt.legend()
		plt.xlabel("r [Mpc]")
		plt.ylabel("km/s")
		plt.savefig(picdir+"pairwise_theory.png",dpi=300)


	f1 = interpolate.interp1d(r, T_pksz_ph1)
	f2 = interpolate.interp1d(r, T_pksz_ph2)
	f = interpolate.interp1d(r, T_pksz)

	return f,f1,f2

#simulation emulator
def get_V_p_zph_temp(r,a,b,rp,vp):
	return vp*(r/rp)**a*np.exp(a*(1-(r/rp)**b)/b)
def covari(s):
	N_r_bin=s.shape[0]
	N_sample=s.shape[1]
	C=np.zeros((N_r_bin,N_r_bin))
	mean=np.mean(s,axis=1)
	for i in range(N_r_bin):
		for j in range(N_r_bin):
			for k in range(N_sample):
				C[i,j]+=(s[i,k]-mean[i])*(s[j,k]-mean[j])
	C=C/N_sample
	return C,np.linalg.inv(C)
def find_ab(s):
	#C,C_=covari(s)
	#print(C_);return 0
	dd=10
	r=np.arange(dd//2,300,dd)
	ll=np.argmin(np.mean(s,axis=1))
	rp=r[ll]
	vp=np.mean(s,axis=1)[ll]

	amin=0.;amax=1.8
	bmin=0.3;bmax=1.7
	cmin=0.;cmax=1.4
	dmin=0.2;dmax=1.3
	a=np.arange(amin,amax,0.02)
	b=np.arange(bmin,bmax,0.02)
	c=np.arange(cmin,cmax,0.02)
	d=np.arange(dmin,dmax,0.02)
	la=len(a)
	lb=len(b)
	lc=len(c)
	ld=len(d)



	r=r.reshape(-1,1,1,1,1)
	a=a.reshape(1,-1,1,1,1)
	b=b.reshape(1,1,-1,1,1)
	c=c.reshape(1,1,1,-1,1)
	d=d.reshape(1,1,1,1,-1)

	V12_temp=get_V_p_zph_temp(r,a,b,rp*d,vp*c)
	V12_simu=(np.mean(s,axis=1).reshape(-1,1,1,1,1))

	chi2=np.sum((V12_simu-V12_temp)**2,axis=0)


	i,j,k,m = np.where(chi2 == chi2.min())
	i=i[0];j=j[0];k=k[0];m=m[0]

	if i==0 or i==la:
		print("a out of range",i)
	if j==0 or j==lb:
		print("b out of range",j)
	if k==0 or k==lc:
		print("c out of range",k)
	if m==0 or m==ld:
		print("d out of range",m)


	return a[0,i,0,0,0],b[0,0,j,0,0],c[0,0,0,k,0],d[0,0,0,0,m],rp,vp
def save_emulator_pairwise_velocity():
	d=np.load("/home/chenzy/code/SZ_planck_DESI/pairwise_ksz/simu_pksz_theory.npz")
	V12=d["V_p"]
	V12_zph=d["V_p_zph"]
	z=d["z"]
	bias=d["b"]
	dd=10
	r=np.arange(dd//2,300,dd)
	a_all=np.zeros(V12_zph.shape[2:4])
	b_all=np.zeros(V12_zph.shape[2:4])
	vp_all=np.zeros(V12_zph.shape[2:4])
	rp_all=np.zeros(V12_zph.shape[2:4])
	for i in range(len(z)):
		for j in range(len(bias[0,:])):
			a,b,c,d,rp,vp=find_ab(V12_zph[:,:,i,j])
			print(a,b,c,d,rp,vp)
			a_all[i,j]=a
			b_all[i,j]=b
			vp_all[i,j]=vp*c
			rp_all[i,j]=rp*d

	np.savez("/home/chenzy/code/SZ_planck_DESI/pairwise_ksz/emulator_pairwise.npz",
         a=a_all,b=b_all,vp=vp_all,rp=rp_all,z=z,bias=bias)
def get_emulator_pairwise_velocity(r,zs,bs):
	#def temp(r,a,b,rp,vp):
	#	return vp*(r/rp)**a*np.exp(a*(1-(r/rp)**b)/b)
	#d=np.load(datadir+"simu_pksz_theory.npz")
	#V12=d["V_p"]
	#V12_zph=d["V_p_zph"]
	#z=d["z"]
	#bias=d["b"]
	d=np.load(datadir+"emulator_pairwise.npz")
	a=d["a"];b=d["b"];rp=d["rp"];vp=d["vp"];z=d["z"];bias=d["bias"]




	zl=np.argmin(np.abs(z[:,0]-zs))
	bl=np.argmin(np.abs(bias[zl,:]-bs))

	#print("=====b,z========")
	#print(zs,bs)
	#print(z[zl,0],bias[zl,bl],zl,bl)
	#print("===============")


	#return np.mean(V12_zph[:,:,zl,bl],axis=1)
	return get_V_p_zph_temp(r, a[zl,bl], b[zl,bl], rp[zl,bl], vp[zl,bl])
def get_emulator_pairwise_velocity_nophotoz(r,zs,bs):
	d=np.load(datadir+"simu_pksz_theory.npz")
	V12=d["V_p"]
	V12_zph=d["V_p_zph"]
	z=d["z"]
	bias=d["b"]

	zl=np.argmin(np.abs(z[:,0]-zs))
	bl=np.argmin(np.abs(bias[zl,:]-bs))

	print("=====b,z========")
	print(zs,bs)
	print(z[zl,0],bias[zl,bl],zl,bl)
	print("===============")

	f = interpolate.interp1d(np.arange(5,300,10), np.mean(V12[:,:,zl,bl],axis=1))

	return f(r)


########################## pipeline ###########################
#1. group data
#range_method="N","M","the"; use N_g
#range_method="condi";		 use the,M
#return Ng,ra,dec,zph,lgM,T_g
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

#2. jackknife pairwise kSZ
#return T_pksz[:,N_bin], n1[:,N_bin], n0[:,N_bin]
def JackKnife(Ng,ra,dec,zph,lgM,Tg,model="corr",alpha=0,dd=20,N_bin=100,N_bin1=10):
	print("JackKnife resampling.....")
	print("Model : ",model)
	print("Mass wight : alpha="+str(alpha))
	r=np.arange(dd//2,300,dd)

	#cut the sky
	the_gal,phi_gal=radec2thephi(ra,dec)
	label_array,label_array_temp=resample_jackknife((phi_gal-0.35)%(2*np.pi),the_gal,N_bin,N_bin1)
	del label_array_temp

	#pairwise
	T_pksz=np.zeros((len(r),N_bin))
	n1=np.zeros((len(r),N_bin))
	n0=np.zeros((len(r),N_bin))
	s_Z2=np.zeros((len(r),N_bin))
	
	for i in tqdm(range(N_bin)):
		if model=="old":
			T_pksz[:,i]=pairwise_old(dec[label_array[i]], ra[label_array[i]],zph[label_array[i]],Tg[label_array[i]],lgM[label_array[i]],alpha,dd)

		if model=="2":
			T_pksz[:,i],n0[:,i]=pairwise_2(dec[label_array[i]], ra[label_array[i]],zph[label_array[i]],Tg[label_array[i]],lgM[label_array[i]],alpha,dd)
		elif model=="3a":
			T_pksz[:,i],n1[:,i]=pairwise_3a(dec[label_array[i]], ra[label_array[i]],zph[label_array[i]],Tg[label_array[i]],lgM[label_array[i]],alpha,dd)

		elif model=="3b":
			(T_pksz[:,i],n1[:,i],n0[:,i]), s_Z2[:, i]=pairwise_3b(dec[label_array[i]], ra[label_array[i]],zph[label_array[i]],Tg[label_array[i]],lgM[label_array[i]],alpha,dd)
		
		elif model=="3c":
			(T_pksz[:,i],n1[:,i],n0[:,i]), s_Z2[:, i]=pairwise_3c(dec[label_array[i]], ra[label_array[i]],zph[label_array[i]],Tg[label_array[i]],lgM[label_array[i]],alpha,dd)

		if model=="corr":
			T_pksz[:,i],n0[:,i]=cal_pairwise_corr(dec[label_array[i]], ra[label_array[i]],zph[label_array[i]],Tg[label_array[i]],lgM[label_array[i]],alpha,dd)
	return T_pksz, n1, n0, s_Z2

#3. jk -> S/N
def signal_and_noise(jk_sample,theory_template,dps=2,n_ev=2):
	t0=time.time()
	r_bin=jk_sample.shape[0]-dps
	N_bin=jk_sample.shape[1]
	s_mean,s_corv=error_esti_jackknife(N_bin,jk_sample)
	C_=(N_bin-r_bin-2.0)/(N_bin-1.)*pseudo_inverse(s_corv[dps:,dps:],n_ev)
	coef_fit,coef_sigma=fit_para(theory_template[dps:],s_mean[dps:],C_)
	chi2_null=chi_square(s_mean[dps:],s_mean[dps:],C_)
	chi2_model=chi_square((s_mean-coef_fit*theory_template)[dps:],(s_mean-coef_fit*theory_template)[dps:],C_)

	return s_mean,s_corv,coef_fit,coef_sigma,chi2_null,chi2_model

'''
Ng,ra,dec,zph,lgM,T_g=Data_pre_processing()
T_pksz, n1, n0=JackKnife(Ng,ra,dec,zph,lgM,T_g)
z_mean=np.mean(zph);bias_mean=np.mean(bias_group(zph,lgM))
dd=20
r=np.arange(dd//2,300,dd)*h
f=get_emulator_pairwise_velocity_nophotoz(r,z_mean,b_mean)
s_mean,s_corv,coef_fit,coef_sigma,chi2_null,chi2_model=signal_and_noise(T_pksz,f)
print(coef_fit/coef_sigma)
'''



