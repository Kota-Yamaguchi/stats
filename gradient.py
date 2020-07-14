import numpy as np
import copy
class Gradient():
#	def steepest_discent(self,func, beta1, beta2,n,p=1,lr=2e-05):	
#		for i in range(n):
#			print("{} times: loglikelihhod {}".format(i, func(beta1, beta2)))
#			d1,d2 = self.Partial_differential(func, beta1, beta2)
#			beta1 +=d1*lr*p
#			beta2 +=d2*lr*p
#			#if d1 < 0.0001 and d2 < 0.0001:
			#	break
			
#		return beta1, beta2
		

		
#	def Partial_differential(self, func, beta1, beta2):
#		h = 0.000001
#		d1 =  (func(beta1+h, beta2)-func(beta1, beta2))/h
#		d2 =  (func(beta1, beta2+h)-func(beta1, beta2))/h
#		return d1, d2


	def steepest_discent(self, func, beta, n,p=1, lr = 2e-05):
		beta_list=[]
		for i in range(n):
			if i % 1000==0:
				print("{}times: loss_value {}".format(i, func(beta)))
				print("beta:{}".format(beta))
			if i % 100 ==0:
				beta_list.append(beta.copy())
			d = self.Partial_differential(func, beta)
			beta += d*lr*p
			
		
		return beta,beta_list
		
	def Partial_differential(self,func, beta):
		h = 1e-04
		d = np.array([])
		for i in range(len(beta)):
			beta_d =copy.deepcopy(beta)
			beta_b = copy.deepcopy(beta)
			beta_d[i] = beta[i]+h
			beta_b[i] = beta[i]-h
			dum = (func(beta_d)-func(beta_b))/(2*h)
			d = np.append(d,dum)
		return d

#	def Partial_differential(self,func, beta):
#		h=np.eye(len(beta))*1e-04
#		beta_d = np.diag(beta)
#		dum = (func(beta_d+h) - func(beta_d-h))/(2*h)
#		d=np.diag(dum)
#		return d

