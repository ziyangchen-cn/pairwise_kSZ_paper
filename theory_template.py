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




