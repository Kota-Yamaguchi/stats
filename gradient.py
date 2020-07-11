

class Gradient():
	def steepest_discent(self,func, beta1, beta2,n):	
		lr = 2e-05
		for i in range(n):
			print("{} times: loglikelihhod {}".format(i, func(beta1, beta2)))
			d1,d2 = self.Partial_differential(func, beta1, beta2)
			beta1 +=d1*lr
			beta2 +=d2*lr
			
		return beta1, beta2
		

		
	def Partial_differential(self, func, beta1, beta2):
		h = 0.000001
		d1 =  (func(beta1+h, beta2)-func(beta1, beta2))/h
		d2 =  (func(beta1, beta2+h)-func(beta1, beta2))/h
		return d1, d2


		
