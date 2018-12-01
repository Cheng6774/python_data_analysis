'''
@Description: python file
@Author: Cheng
@Github: https://github.com/Cheng6774
@Date: 2018-11-23 17:09:56
@LastEditors: Cheng
@LastEditTime: 2018-11-23 23:48:18
'''

#5.1
import numpy as np
import pandas as pd

np.random seed(42)
a = np.random.randn(3,4)
a[2][2] = np.nan
print(a)

np.savetxt('np.csv',a,fmt='%.2f',delimiter=',',header="#1,#2,#3,#4")
df=pd.DataFrame(a)
print(df)
df.to_csv('pd.csv',float_format='%.2f',na_rep="NAN!")

#5.2
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile
from os.path import getsize

np.random.seed(42)
a = np.random.randn(365,4)

tmpf = NamedTemporaryFile()
np.savetxt(tmpf,a,delimiter=',')
print("Size CSV file",getsize(tmpf.name))
tmpf = NamedTemporaryFile()
np.save(tmpf,a)
tmpf.seak(0)
loaded = np.load(tmpf)
print("Shape",loaded.shape)
print("Size .npy file",getsize(tmpf.name))




