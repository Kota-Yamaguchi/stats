
from scipy.stats import poisson
from scipy.stats import binom
from enum import Enum

class Probability(Enum):
	poisson = 1
	binominal = 2


class Probability_select():
	#入力は必ずEnumの形で入力する、Int型での比較はできない
	def __init__(self, ty=Probability.poisson):		
	
		if Probability.poisson == ty:
			self.prob = lambda x,mean:self.poisson_curve(x,mean)
		elif Probability.binominal == ty:
			self.prob = lambda k,N,p:self.binom_curve(k,N,p)
	


	def binom_curve(self,k , p, N):

		f = binom.pmf(k, N ,p)
		return f	

	def poisson_curve(self, k, mean=2):
		try:
			print(mean)
			f = poisson.pmf(k,mu = mean)
			return f
		
		except RuntimeError:
			print("Error")
