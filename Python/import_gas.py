import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt

#my_data = genfromtxt('../Data/natgas.data.csv', delimiter=',', 
#                     dtype="i8,i8,i8,i8,f8,f8,f8,f8,f8,i8");
#                     names= ['VarName1', 'STOCKS', 'HDD_FORE', 'CDD_FORE', 'JFKTEMP', 'CLTTEMP', 'ORDTEMP', 'HOUTEMP', 'LAXTEMP', 'NXT_CNG_STK'])


my_data = genfromtxt('../Data/natgas.data.csv', delimiter=',',dtype='f8')
my_data = my_data[1:,]

target = my_data[:,9]


plt.title("Matplotlib demo") 
plt.xlabel("Sample index") 
plt.ylabel("Next week change in natgas") 
plt.plot(target) 
plt.show()