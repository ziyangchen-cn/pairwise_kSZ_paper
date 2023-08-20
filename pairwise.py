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

from funcs import *

'''
"alpha": mass weight
"dd": length of the radius bin, Mpc/h
'''


def pairwise_calculate(dec_rad, ra_rad, redshift, T_ap, lgM, alpha, dd):

	dec_deg = dec_rad * np.pi/180
	ra_deg  = ra_rad * np.pi/180
	d_coming =z2dc(redshift)
	
	cosDec=np.cos(dec_deg)
	sinDec=np.sin(dec_deg)
	cosRa=np.cos(ra_deg)
	sinRa=np.sin(ra_deg)
	coor_x=d_coming*cosDec*cosRa
	coor_y=d_coming*cosDec*sinRa
	coor_z=d_coming*sinDec

	M=10**(lgM*alpha)/np.mean(10**(lgM*alpha))/2

	Cij = np.zeros(300//dd)
	Cij2 = np.zeros(300//dd)

	Zij = np.zeros(300//dd)
	Zij2 = np.zeros(300//dd)
	Zij3 = np.zeros(300//dd)
	Zij4 = np.zeros(300//dd)

	CijZij = np.zeros(300//dd)
	CijZij2 = np.zeros(300//dd)

	Tij = np.zeros(300//dd)
	TijCij = np.zeros(300//dd)
	TijZij = np.zeros(300//dd)
	TijZij2 = np.zeros(300//dd)

	N_pair = np.zeros(300//dd)
	MiMj = np.zeros(300//dd)

	#T_ap, coor_x, coor_y, coor_z, cosDec, cosRa, sinDec, sinRa, M, d_coming, redshift, 
	@jit(nopython=True)
	def pairwise_for(Cij, Cij2, Zij, Zij2, Zij3, Zij4, CijZij, CijZij2, Tij, TijCij, TijZij, TijZij2, N_pair, MiMj):
		for i in (range(len(T_ap))):
			for j in range(i+1,len(T_ap)):
				r=np.sqrt((coor_x[i]-coor_x[j])**2+(coor_y[i]-coor_y[j])**2+(coor_z[i]-coor_z[j])**2)
				if r>300:
					continue
				label_r=int(r/dd)
				cos=cosDec[i]*cosDec[j]*cosRa[i]*cosRa[j]+cosDec[i]*cosDec[j]*sinRa[i]*sinRa[j]+sinDec[i]*sinDec[j]

				cij=(M[i]+M[j])*(d_coming[i]-d_coming[j])*(1+cos)/2/np.sqrt(d_coming[i]**2+d_coming[j]**2-2*d_coming[i]*d_coming[j]*cos)

				zij = redshift[i] - redshift[j]
				tij = T_ap[i] - T_ap[j]

				Cij[label_r]+=cij
				Cij2[label_r]+=cij**2

				Zij[label_r]+=zij
				Zij2[label_r]+=zij**2
				Zij3[label_r]+=zij**3
				Zij4[label_r]+=zij**4

				CijZij[label_r]+=cij*zij
				CijZij2[label_r]+=cij*zij**2

				Tij[label_r]+=tij
				TijCij[label_r]+=tij*cij
				TijZij[label_r]+=tij*zij
				TijZij2[label_r]+=tij*zij**2

				N_pair[label_r]+=1
				MiMj[label_r]+=(M[i]+M[j])

		print(r, N_pair)

	pairwise_for(Cij, Cij2, Zij, Zij2, Zij3, Zij4, CijZij, CijZij2, Tij, TijCij, TijZij, TijZij2, N_pair, MiMj)
	print(N_pair)
	Equation_M = np.zeros((4,4, len(N_pair)))
	Equation_A = np.zeros((4, len(N_pair)))
	Equation_M[ 0, 0, :] = Cij2/N_pair
	Equation_M[ 0, 1, :] = CijZij2/N_pair
	Equation_M[ 0, 2, :] = CijZij/N_pair
	Equation_M[ 0, 3, :] = Cij/N_pair

	Equation_M[ 1, 1, :] = Zij4/N_pair
	Equation_M[ 1, 2, :] = Zij3/N_pair
	Equation_M[ 1, 3, :] = Zij2/N_pair
   

	Equation_M[ 2, 2, :] = Zij2/N_pair
	Equation_M[ 2, 3, :] = Zij/N_pair
	
	Equation_M[ 3, 3, :] = 1

	Equation_A[0, :] = TijCij/N_pair
	Equation_A[1, :] = TijZij2/N_pair
	Equation_A[2, :] = TijZij/N_pair
	Equation_A[3, :] = Tij/N_pair
	MiMj = MiMj/N_pair

	delta_redshift = np.sqrt(Zij2/N_pair)

	return Equation_M, Equation_A, delta_redshift, MiMj


	

	
