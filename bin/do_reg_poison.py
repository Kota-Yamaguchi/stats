from static_method import Sampling
import matplotlib.pyplot as plt
import numpy as np

sample = Sampling()
data = sample.generate_poisson_Reg()
#data = sample.generate_binominal_Reg()
#data = data.iloc[:,0:2]
#a = sample.likelihood_presume(data)#,family=2,link="logit")
a = sample.fitting(data,family=1,link="log",n=3000,loss_function = "LSE")
#a = sample.LSE(data)
x = data[0]
y = a(x)
data_pre = np.append(np.array([x]),np.array([y]),axis=0)
data_pre = sample.trans_pd2D(data_pre)
data = sample.destandalization(data)
data_pre=sample.destandalization(data_pre)	
plt.scatter(data_pre[0],data_pre[1],color="r")
plt.scatter(data[0], data[1])
plt.show()
