from static_method import Sampling
import matplotlib.pyplot as plt 

sample = Sampling()
data = sample.generate_binominal_Reg()
a,b,r = sample.fitting(data,family=2,link="logit",loss_function="MLE",n=7000,lr=4e-05)

sample.animation(data[0], b)
#a = sample.MLE(data1)
print(a(data[0]))	
x = data[0]
plt.scatter(x,a(x)*10,color="r")
plt.scatter(x, data[1])
plt.show()
