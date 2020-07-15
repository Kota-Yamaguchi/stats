import numpy as np
from enum import Enum


class Link(Enum):
	log = 1
	logit = 2 
#	identity = 3


class Link_select():
	def __init__(self,ty):
		if ty == Link.log:
			self.link = lambda x: self.log(x)
#		elif ty == Link.identify:
#			self.link = lambda x: self.identity(x)
		elif ty == Link.logit:
			self.link = lambda x: self.logit(x)
	def log(self, x):
		return np.exp(x)
	
#	def identity(self,x):
#		return x
	
	def logit(self,x):
		return 1/(1+np.exp(x))	
		
		
