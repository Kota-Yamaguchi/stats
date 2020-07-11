import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy.stats import poisson
from scipy.stats import binom
import statsmodels.api as sm
from Probability import Probability
from Probability import Probability_select
from link import Link
from link import Link_select
from gradient import Gradient


class Sampling():
	def generate_poisson(self, lamb = 2, size=1000):
		x = np.random.poisson(lamb, size)
		#array = np.arange(len(x))+1
		return x#,array
	def generate_poisson_Reg(self, Min= 0, Max=10, n=100):
		data = np.array([])
		x = np.random.uniform(Min, Max, n)
		for i in x:
			particle_data =  self.generate_poisson(i, 1)
			data=np.append(data, particle_data,axis=0)
		return x, data
			
	def generate_binominal(self, prob=0.5, l=20, size = 100):
		f = binom.rvs(l,prob ,size=size)
		return f

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
			data = pd.read_csv(filename, encoding="SHIFT-JIS")
			return data

	def trans_pd1D(self, array, columns="number"):
		pddata = pd.DataFrame({columns:array})
		return pddata
	def trans_pd2D(self, array):
		pddata = pd.DataFrame(array.T)
		return pddata
	#def gauss_curve(self, k, mean = 2, var = 1.0):
	#	return a

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
	
	def likelihood_presume(self,data,link="log",family=1,n=10000):
		prob_dict = {1:Probability.poisson, 2: Probability.binominal }
		p =  Probability_select(prob_dict[family])		
		prob = p.prob
		link_dict = {"log":Link.log, "identity": Link.identity, "logit":Link.logit}
		l = Link_select(link_dict[link])
		link = l.link
		beta1 = np.random.random()
		beta2 = np.random.random()
		beta2=0
		if family ==1:
			print("data{}".format(data))
			print(link(data*beta2+beta1))
			print(prob(data, link(data*beta2+beta1)))
			predicter = lambda b1,b2 : self.likelihood_function(self.poisson_curve(data, link(data*b2+b1)))
			grad = Gradient()
			bata1 , beta2 =grad.steepest_discent(predicter, beta1,beta2,n=10000)
		
		a = lambda x : link(x*beta2 + beta1)
		print("beta1:{} beta2:{}".format(beta1,beta2))
		return a
					
	
	def maximize_poisson_likelihood(self, data):
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
		

if __name__=="__main__":
	
	sample = Sampling()
	x, data = sample.generate_poisson_Reg()
	print(data)
	a = sample.likelihood_presume(data)
	print(a(x))	
	plt.scatter(x,a(x),color="r")
	plt.scatter(x, data)
	plt.show()
