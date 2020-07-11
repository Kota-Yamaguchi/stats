import numpy as np
from Probability import Probability
from Probability import Probability_select


class MCMC():
	def __init__(self, ty=1):
		prob_dict = {1:Probability.poisson, 2: Probability.binominal }
		p =  Probability_select(prob_dict[ty])		
		self.prob = p.prob
	def metroPolis_poisson(self,data,score_function,sample_number = 10000):
		hist = []
		mean = np.random.randint(min(data), max(data))
		probs_ini = self.prob(data, mean)
		score = score_function(probs_ini)
		print("initial likelihood:{}".format(score))
		for i in range(sample_number):
			incr = [0.03,-0.03]
			select = np.random.choice(incr)
			mean_next = mean + select
			if (mean_next<1):
				mean_next=1
			probs = self.prob(data, mean_next)
			next_score = score_function(probs)
			print("{}times Likelihood:{}".format(i,next_score))
			if (next_score > score):
				mean = mean_next
				print("probability 1")
				score = next_score
			else:
				boo = np.exp((next_score-score))
				boolean = [True,False]
				decision = np.random.choice(boolean,p=[boo, 1-boo])
				
				print("probability {}".format(boo))
				if decision:
					mean = mean_next
					score = next_score
			hist.append(mean)
		
		return hist

	def metroPolis_binom(self, data, score_function,N=20, sample_number=100000):
		hist = []
		p = np.random.random()
		probs_ini = self.prob(data, p, N)
		score = score_function(probs_ini)
		print("initial likelihood:{}".format(score))
		for i in range(sample_number):
			incr =[0.01, -0.01]
			select = np.random.choice(incr)
			p_next = p+ select
			if (p_next>1):
				p_next=1
			elif (p_next<0):
				p_next=0
			probs = self.prob(data, p_next, N)
			next_score = score_function(probs)
			print("{}times Likelihood:{}".format(i,next_score))
			if (next_score > score):
				p = p_next
				print("probability 1")
				score = next_score
			else:
				boo = np.exp((next_score-score))
				boolean = [True,False]
				decision = np.random.choice(boolean,p=[boo, 1-boo])
			
				print("probability {}".format(boo))
				if decision:
					p = p_next
					score = next_score
			hist.append(p)
		
		return hist

	#def train(self, data):
	#	like = lambda c:sample.likelihood_function(c)


if __name__ == "__main__":
	from generate_poison import Sampling
	import matplotlib.pyplot as plt
	sample = Sampling()
	fi = sample.csv_reader("sample.csv")
	for i in range(len(fi)):
		for k in range(len(fi[i])):
			fi[i][k] = int(fi[i][k])
	like = lambda c:sample.likelihood_function(c)
	mcmc = MCMC(2)
	hist = mcmc.metroPolis_binom(data=fi[1], score_function=like)
	#hist = mcmc.metroPolis_poisson(data=fi[1],score_function=like)
	mean = np.mean(hist)
	print(mean)	
	plt.hist(hist[1000:])
	
	plt.show()
