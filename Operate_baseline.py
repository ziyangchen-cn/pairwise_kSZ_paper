
from data_preparing import *
from pairwise import *
from jackknife import *


g_posi="SGC"
cmbtype="217"
alpha=1
range_method="N"
dd=20
N_jk=100
ap_the=30
gv="DR8"

filename="Equation_M_A_"+gv+"_"+g_posi+"_"+cmbtype+"_"+range_method+"_ap"+str(ap_the)+"_a"+str(alpha)+"_dd"+str(dd)+"_Njk"+str(N_jk)+".npz"

if g_posi=="all":
	Ngr=[];rar=[]; decr=[]; zphr=[]; lgMr=[]; T_gr=[]
	for i in ["NGC","SGC"]:
		Ng, ra, dec, zph, lgM, T_g = Data_pre_processing(gv=gv,g_posi=i,cmbtype=cmbtype,range_method=range_method,ap_the=ap_the/10.0)
		Ngr.append(Ng)
		rar.append(ra)
		decr.append(dec)
		zphr.append(zph)
		lgMr.append(lgM)
		T_gr.append(T_g)
	Ng=np.hstack(Ngr)
	ra=np.hstack(rar)
	dec=np.hstack(decr)
	zph=np.hstack(zphr)
	lgM=np.hstack(lgMr)
	T_g=np.hstack(T_gr)
	N_bin=N_jk*2; N_bin1=int(np.sqrt(N_jk)+0.1)
else:
	Ng, ra, dec, zph, lgM, T_g = Data_pre_processing(gv=gv,g_posi=g_posi,cmbtype=cmbtype,range_method=range_method,ap_the=ap_the/10.0)
	N_bin=N_jk; N_bin1=int(np.sqrt(N_jk)+0.1)
z_mean = np.mean(zph)
M_mean = np.mean(10**lgM)

#cut the sky
the_gal,phi_gal=radec2thephi(ra,dec)
label_array,label_array_temp=resample_jackknife((phi_gal-0.35)%(2*np.pi),the_gal,N_bin,N_bin1)
del label_array_temp

#pairwise for each jackknife sample
Equation_M = np.zeros(( 4, 4,len(np.arange(dd//2,300,dd)), N_bin))
Equation_A = np.zeros(( 4,len(np.arange(dd//2,300,dd)), N_bin))
delta_redshift= np.zeros(( len(np.arange(dd//2,300,dd)), N_bin))
for i in tqdm(range(N_bin)):
    Equation_M[:,:,:,i], Equation_A[:,:,i], delta_redshift[:,i] = pairwise_calculate(dec_rad=dec[label_array[i]], ra_rad=ra[label_array[i]], redshift=zph[label_array[i]], T_ap=T_g[label_array[i]], lgM=lgM[label_array[i]], alpha=alpha, dd=dd)

np.savez(filename, Equation_M=Equation_M, Equation_A=Equation_A, delta_redshift=delta_redshift, z_mean = z_mean, M_mean = M_mean) 
