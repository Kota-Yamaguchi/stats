import numpy as np
import math as mt
import matplotlib.pyplot as plt 



def PCA(data):

 cov=np.empty((0,traj.n_atoms*3,traj.n_atoms*3),int)
 for i in range(traj.n_frames):
  a=np.array([xyz[i].reshape(traj.n_atoms*3),xyz_mean])
  a=a.T
  cov=np.append(cov,np.array([np.cov(a)]),axis=0)

 cov_mean=np.mean(cov,axis=0)

 eig_val,eig_vec=np.linalg.eig(cov_mean)
 Z1=np.array([])
 Z2=np.array([])
 for i in range(traj.n_frames):
  a=np.dot(xyz[i].reshape(traj.n_atoms*3)-xyz_mean,eig_vec[0])
  b=np.dot(xyz[i].reshape(traj.n_atoms*3)-xyz_mean,eig_vec[1])
  Z1=np.append(Z1,a)
  Z2=np.append(Z2,b)
 
 fs=traj.n_frames
 z1z2=np.array([Z1,Z2])
 np.save("output_pc1",z1z2)
 np
 return Z1,Z2,fs
#ax1=plt.subplot()


#plt.plot(Z1,Z2)

#plt.show()

