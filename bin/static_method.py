import numpy as np
import matplotlib.animation as animation
import scipy.stats as st
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy.stats import poisson
from scipy.stats import binom
from Probability import Probability
from Probability import Probability_select
from link import Link
from link import Link_select
from gradient import Gradient


class Sampling():
	std_preserve = []
	mean_preserve = []
	link =0
	def generate_poisson(self, lamb = 2, size=1000):
		x = np.random.poisson(lamb, size)
		#array = np.arange(len(x))+1
		return x#,array
	def generate_poisson_Reg(self, Min= 2, Max=20, n=100):
		data = np.array([])
		x = np.random.uniform(Min, Max, n)
		for i in x:
			particle_data =  self.generate_poisson(i, 1)
			data=np.append(data, particle_data,axis=0)
	
		data = np.append(np.array([x]),np.array([data]),axis=0)
		data = self.trans_pd2D(data)
		return data
			
	def generate_binominal(self, prob=0.5, limit=20, size = 100):
		f = binom.rvs(limit,prob ,size=size)
		return f

	def generate_binominal_Reg(self, Min= 7,Max = 12,limit=10, size=1):
		data = np.array([])
		ft = np.array([])
		x = np.random.uniform(Min, Max,200)
		limit_array = np.array([limit]*len(x))
		q = lambda z,f : 1/(1+np.exp(-(-19.5+1.95*z+2.02*f)))
		for i in x:
			
			f = np.random.choice([0,1])
			particle_data = self.generate_binominal(prob=q(i,f),limit=limit,size=size)
			data = np.append(data, particle_data,axis=0)
			ft = np.append(ft,[1 if f ==1 else 0])
		data = np.append(np.array([x]),np.array([data]),axis=0)
		data = np.append(data, np.array([limit_array]),axis=0)
		data = np.append(data, np.array([ft]),axis=0)
		data = self.trans_pd2D(data)
		return data
		

	def csv_writer(self, x ,filename="sample.csv",index=False,header=True,ty=1):
		if ty==0:
			row1 = x[0]
			row2 = x[1]
			with open(filename, "w") as f:
				w = csv.writer(f)
				w.writerows([row1,row2])
		if ty==1:
			print("pandas形式")
			x.to_csv(filename,header=header,index=index,encoding = "shift-jis")				
		
		
	def csv_reader(self, filename, ty=0):
		if ty==0:
			x = []
			with open(filename ,"r") as f:
				reader = csv.reader(f) 
				for r in reader:
					x.append(r)
			return x				
		elif ty==1:
			data = pd.read_csv(filename, encoding="shift-jis")
			return data

	def trans_pd1D(self, array, columns="number"):
		pddata = pd.DataFrame({columns:array})
		return pddata
	def trans_pd2D(self, array, ):
		pddata = pd.DataFrame(array.T)
		return pddata
	#def gauss_curve(self, k, mean = 2, var = 1.0):
	#	return a

	def standalization(self, data):
		self.mean_preserve =[]
		self.std_preserve =[]
		for i in range(data.shape[1]):
			self.mean_preserve.append(data[i].mean())
			self.std_preserve.append(data[i].std())
			data[i]= (data[i]-data[i].mean())/data[i].std()
		return data			
	def destandalization(self, data):
		for i in range(data.shape[1]):
			data[i] = data[i]*self.std_preserve[i] + self.mean_preserve[i]
		return data		


	def binom_curve(self,k , p, N):

		f = binom.pmf(k, N ,p)
		return f	

	def poisson_curve(self, k, mean=2):
		f = poisson.pmf(k,mu = mean)
		return f 
	
	def likelihood_function(self, prob):
		log_probs = np.log(prob)
		log_likelihood = np.sum(log_probs)
		return log_likelihood
	def _Coef_deter(self, func, data):
#		stnd = self.standalization(data)
#		print(np.sum(stnd[0]))
		y = data[1]
		y_p = func(data[0])
		all_variance = np.sum((y-np.mean(y))**2)
		reg_variance = np.sum((y_p-np.mean(y))**2)
		Cofdeterminant = reg_variance/all_variance
		return Cofdeterminant
	
	def fitting(self,data,link="log",family=1,n=10000, loss_function="MLE",lr= 4e-08):
		prob_dict = {1:Probability.poisson, 2: Probability.binominal }
		p =  Probability_select(prob_dict[family])		
		prob = p.prob
		link_dict = {"log":Link.log,"logit":Link.logit}# "identity": Link.identity, "logit":Link.logit}
		l = Link_select(link_dict[link])
		link = l.link
		beta1 = np.random.random()
		beta2 = np.random.random()
		#beta2=0
		beta = np.array([beta1, beta2])
		if family ==1:
			prob_func = lambda b : self.poisson_curve(data[1], link(np.multiply(data[0],b[1])+b[0]))
		if family == 2:
			prob_func = lambda b : self.binom_curve(data[1], link(data[0]*b[1]+b[0]), N=10)	
		if loss_function=="MLE":
			p=1
			lr = lr
			predicter = lambda b : self.likelihood_function(prob_func(b))
		elif loss_function=="LSE":
			lr = 5e-04
			p=-1
			data = self.standalization(data)
			print(data)
			predicter = lambda b : np.sum((data[1]-link(data[0]*b[1]+b[0]))**2)
		grad = Gradient()
		bata,betalist =grad.steepest_discent(predicter, beta,n=n, p=p,lr=lr)
		a = lambda x : link(x*beta[1] + beta[0])
		r = self._Coef_deter(a, data)
		self.link = lambda x,b : link(x*b[1] + b[0])
		print("beta1:{} beta2:{}".format(beta[0],beta[1]))
		return a, betalist, r
	

	def _maximize_poisson_likelihood(self, data):
		log_likelihood = -10000000 
		i = 0.1
		probs = self.poisson_curve(data, i)
		likelihood_next = self.likelihood_function(probs)
		#log_probs = np.log(probs)
		#log_likelihood_next = np.sum(log_probs)
		i+=0.1
		while(log_likelihood_next > log_likelihood):  			
			print("平均値：{}".format(i))
			log_likelihood = log_likelihood_next
			
			probs = self.poisson_curve(data, i)
			log_probs = np.log(probs)
			log_likelihood_next = np.sum(log_probs)
			print("{}".format(log_likelihood_next))
			if log_likelihood_next > log_likelihood:
				i+=0.1
		return log_likelihood, i

	def animation(self,x, blist, stdz="off"):
		ims = []
		fig = plt.figure()
		if stdz== "off":
			for i in range(len(blist)):
				y = self.link(x,blist[i]) 
				plt.scatter(x,y)
				plt.pause(0.3)
		elif stdz=="on":
			for i in range(len(blist)):
				y = self.link(x,blist[i])
				data_pre = np.append(np.array([x]),np.array([y]),axis=0)
				data_pre = self.trans_pd2D(data_pre)
				data_pre=self.destandalization(data_pre)
				plt.scatter(data_pre[0],data_pre[1])
				plt.pause(0.3)
			#ims.append(im)
		#ani = animation.ArtistAnimation(fig, ims, interval=1)
		plt.show()

			

if __name__=="__main__":
	
	sample = Sampling()
	data = sample.generate_poisson_Reg()
	print(data)
	aMLE,beta_list,r = sample.fitting(data,family=1,link="log",n=10000,loss_function = "MLE")
	print("決定係数:{}".format(r))
	sample.animation(data[0], beta_list)
	aLSE,beta_list,r = sample.fitting(data,family=1,link="log",n=10000,loss_function = "LSE")
	#a = sample.LSE(data)
	print("決定係数:{}".format(r))
	x = data[0]
	sample.animation(data[0], beta_list,stdz="on")
	yLSE = aLSE(x)

	data_pre = np.append(np.array([x]),np.array([yLSE]),axis=0)
	data_pre = sample.trans_pd2D(data_pre)
	data = sample.destandalization(data)
	data_pre=sample.destandalization(data_pre)
	print(data_pre)	
	yMLE = aMLE(x)
	print("MSE:{}".format(np.sum(((yMLE -data[1])**2))))
	plt.scatter(data[0], yMLE,color="g")
	plt.scatter(data[0],data_pre[1],color="r")
	#plt.scatter(data[0],yMLE,color="g")
	#lt.scatter(data[0],yLSE(x),color="r")
	plt.scatter(data[0], data[1])
	plt.show()
