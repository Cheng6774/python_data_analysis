'''
@Description: python file
@Author: Cheng
@Github: https://github.com/Cheng6774
@Date: 2018-11-18 17:41:19
@LastEditors: Cheng
@LastEditTime: 2018-11-23 22:42:53
'''

import numpy as np
from scipy.stats import scoreatpercentile
import pandas as pd

data = pd.read_csv("co2.csv", index_col=0, parse_dates=True)

co2 = np.array(data.co2)

print("The statistical valus for amounts of co2 in atmosphere : \n")
print("Max method : ", co2.max())
print("Max function : ", np.max(co2))

print("Min method : ", co2.min())
print("Min function : ", np.min(co2))

print("Mean method : ", co2.mean())
print("Mean function : ", np.mean(co2))

print("Std method : ", co2.std())
print("Std function : ", np.std(co2))

print("Median : ", np.median(co2))
print("Score at percentile 50 : ", scoreatpercentile(co2, 50))

# 线性代数
A = np.mat("2 4 6 ; 4 2 6 ; 10 -4 18")
print("A\n", A)

inverse = np.linalg.inv(A)
print("inverse of A \n", inverse)
print("Check \n", A * inverse)
print("Error \n", A * inverse - np.eye(3))

# 解线性方程组
A = np.mat("2 4 6;4 2 6;10 -4 18")
print("A\n", A)
b = np.array([0, 8, -9])
print("b\n", b)

x = np.linalg.solve(A, b)
print("Solution", x)
print("Check\n", np.dot(A, x))

# 计算特征值和特征向量
A = np.mat("3 -2;1 0")
print("A\n", A)

print(("Eigenvalues", np.linalg.eig(A)))

eigenvalues, eigenvectors = np.linalg.eig(A)
print("First tuple of eig", eigenvalues)
print("Second tuple of eig \n", eigenvectors)
for i in range(len(eigenvalues)):
    print("Left", np.dot(A, eigenvectors)[:, i])
    print("Right", eigenvalues[i] * eigenvectors[:, i])

# 用二项式分布进行博弈
import numpy as np
from matplotlib.pyplot import plot, show

cash = np.zeros(10000)
cash[0] = 1000
outcome = np.random.binomial(9, 0.5, size=(len(cash)))
for i in range(1, len(cash)):
    if outcome[i] < 5:
        cash[i] = cash[i - 1] - 1
    elif outcome[i] < 10:
        cash[i] = cash[i - 1] + 1
    else:
        raise AssertionError("Unexpected outcome" + outcome)
print(outcome.min(), outcome.max())
# plot(np.arange(len(cash)),cash)
#show()

#正态分布采样
# import numpy as np
# import matplotlib.pyplot as plt

# N=10000

# normal_values = np.random.normal(size=N)
# dummy, bins, dummy = plt.hist(normal_values, int(np.sqrt(N)), normed=True, lw=1)
# sigma = 1
# mu = 0
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),lw=2)
# show()

# 用scipy进行正态检验
import numpy as np
from scipy.stats import shapiro
from scipy.stats import anderson
from scipy.stats import normaltest

flutrends = np.loadtxt(
    "goog_flutrends.csv",
    delimiter=',',
    usecols=(1, ),
    skiprows=1,
    converters={1: lambda s: float(s or 0)},
    unpack=True)
N = len(flutrends)
normal_values = np.random.normal(size=N)
zero_values = np.zeros(N)

print("Normal Values Shapiro", shapiro(normal_values))
#print("Zeroes Shapiro", shapiro(zero_values))
print("Flu Shapiro", shapiro(flutrends))

print("Normal Values Anderson", anderson(normal_values))
#print("Zeroes Anderson", anderson(zero_values))
print("Flu Anderson", anderson(flutrends))

print("Normal Values normaltest", normaltest(normal_values))
#print("Zeroes normaltest", normaltest(zero_values))
print("Flu normaltest", normaltest(flutrends))

#创建掩码式numpy数组
import numpy
import scipy
import matplotlib.pyplot as plt

face = scipy.misc.face()

random_mask = numpy.random.randint(0, 2, size=face.shape)

plt.subplot(221)
plt.title("Original")
plt.imshow(face)
plt.axis('off')

masked_array = numpy.ma.array(face, mask=random_mask)

plt.subplot(222)
plt.title("Masked")
plt.imshow(masked_array)
plt.axis('off')

plt.subplot(223)
plt.title("Log")
plt.imshow(numpy.ma.log(face).astype("float32"))
plt.axis('off')

plt.subplot(224)
plt.title("Log Masked")
plt.imshow(numpy.ma.log(masked_array).astype("float32"))
plt.axis('off')

plt.show()

# 忽略负值和极值
import numpy as np
from datetime import date
import sys
import matplotlib.pyplot as plt

salary = np.loadtxt(
    "MLB2008.csv", delimiter=',', usecols=(1, ), skiprows=1, unpack=True)
triples = np.arange(0, len(salary), 3)
print("Triples", triples[:10], "...")

signs = np.ones(len(salary))
print("Signs", signs[:10], "...")

signs[triples] = -1
print("Signs", signs[:10], "...")

ma_log = np.ma.log(salary * signs)
print("Masked logs", ma_log[:10], "...")

dev = salary.std()
avg = salary.mean()
inside = np.ma.masked_outside(salary, avg - dev, avg + dev)
print("Inside", inside[:10], "...")

plt.subplot(311)
plt.title("Original")
plt.plot(salary)

plt.subplot(312)
plt.title("Log Masked")
plt.plot(np.exp(ma_log))

plt.subplot(313)
plt.title("Not Extreme")
plt.plot(inside)

plt.subplots_adjust(hspace=.9)

plt.show()
