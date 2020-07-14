
from static_method import Sampling
import matplotlib.pyplot as plt
import numpy as np

sample = Sampling()
data = sample.generate_poisson_Reg()
a,beta_list = sample.fitting(data,family=1,link="log",n=3000,loss_function = "MLE")
x = data[0]
plt.scatter(x,a(x),color="r")
plt.scatter(x, data[1])
plt.show()
