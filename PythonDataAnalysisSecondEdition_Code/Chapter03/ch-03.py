'''
@Description: -*- coding: utf-8 -*-
@Author: Cheng
@Github: https://github.com/Cheng6774
@Date: 2018-11-18 16:20:33
@LastEditors: Please set LastEditors
@LastEditTime: 2018-11-22 23:58:30
'''


import numpy as np
import scipy as sp
import pandas as pd

#######################
from pandas.io.parsers import read_csv

df = read_csv("WHO_first9cols.csv")
print("Dataframe:\n", df)

print("Shape:\n", df.shape)
print("\n")
print("Length:\n", len(df))
print("\n")
print("Column Headers:\n", df.columns)
print("\n")
print("Data types:\n", df.dtypes)
print("\n")
print("Index:\n", df.index)
print("\n")
print("Values:\n", df.values)

country_col = df["Country"]
print("Type df:\n", type(df), "\n")
print("Type country col:\n", type(country_col), "\n")

print("Last 2 countries:\n", country_col[-2:], "\n")
print("Last 2 countries type:\n", type(country_col[-2:]), "\n")

print("Series shape:\n", country_col.shape, "\n")
print("Series index:\n", country_col.index, "\n")
print("Series values:\n", country_col.values, "\n")
print("Series name:\n", country_col.name, "\n")

print("Last 2 countries:\n", country_col[-2:], "\n")
print("Last 2 countries type:\n", type(country_col[-2:]), "\n")

last_col = df.columns[-1]
print("Last df column signs:\n", last_col, np.sign(df[last_col]), "\n")

np.sum([0, np.nan])
df.dtypes
print(np.sum(df[last_col] - df[last_col].values))

#############################
import quandl
sunspots = quandl.get("SIDC/SUNSPOTS_A")
print("Head 2:\n", sunspots.head(2))
print("Tail 2:\n", sunspots.tail(2))

last_date = sunspots.index[-1]
print("Last value:\n", sunspots.loc[last_date])

print("Values slice by date:\n", sunspots["20020101":"20131231"])

print("Slice from a list of indices:\n", sunspots.iloc[[2, 4, -4, -2]])

print("Scalar with Iloc:", sunspots.iloc[0, 0])
print("Scalar with iat", sunspots.iat[1, 0])

print("Boolean selection:\n", sunspots[sunspots > sunspots.mean()])

print(
    "Boolean selection with column label:\n",
    sunspots[sunspots['Number of Observations'] >
             sunspots['Number of Observations'].mean()])

############################
import quandl

# Data from http://www.quandl.com/SIDC/SUNSPOTS_A-Sunspot-Numbers-Annual
# PyPi url https://pypi.python.org/pypi/Quandl
sunspots = quandl.get("SIDC/SUNSPOTS_A")
print("Describe", sunspots.describe(), "\n")
print("Non NaN observations", sunspots.count(), "\n")
print("MAD", sunspots.mad(), "\n")
print("Median", sunspots.median(), "\n")
print("Min", sunspots.min(), "\n")
print("Max", sunspots.max(), "\n")
print("Mode", sunspots.mode(), "\n")
print("Standard Deviation", sunspots.std(), "\n")
print("Variance", sunspots.var(), "\n")
print("Skewness", sunspots.skew(), "\n")
print("Kurtosis", sunspots.kurt(), "\n")

############################
import pandas as pd
from numpy.random import seed
from numpy.random import rand
from numpy.random import randint
import numpy as np

seed(42)

df = pd.DataFrame({
    'Weather': ['cold', 'hot', 'cold', 'hot', 'cold', 'hot', 'cold'],
    'Food':
    ['soup', 'soup', 'icecream', 'chocolate', 'icecream', 'icecream', 'soup'],
    'Price':
    10 * rand(7),
    'Number':
    randint(1, 9)
})

print(df)
weather_group = df.groupby('Weather')
i = 0
for name, group in weather_group:
    i = i + 1
    print("Group", i, name)
    print(group)

print("Weather group first\n", weather_group.first())
print("Weather group last\n", weather_group.last())
print("Weather group mean\n", weather_group.mean())

wf_group = df.groupby(['Weather', 'Food'])

print("WF Aggregated\n", wf_group.agg([np.mean, np.median]))

print("df :3\n", df[:3])
print("Concat Back together\n", pd.concat([df[:3], df[3:]]))
print("Appending rows\n", df[:3].append(df[5:]))

#######################
dests = pd.read_csv('dest.csv')
print("Dests\n", dests)

tips = pd.read_csv('tips.csv')
print("Tips\n", tips)

print("Merge() on key\n", pd.merge(dests, tips, on='EmpNr'))
print("Dests join() tips\n", dests.join(tips, lsuffix='Dest', rsuffix='Tips'))

print("Inner join with merge()\n", pd.merge(dests, tips, how='inner'))
print("Outer join\n", pd.merge(dests, tips, how='outer'))

df = pd.read_csv('WHO_first9cols.csv')
# Select first 3 rows of country and Net primary school enrolment ratio male (%)
df = df[['Country', df.columns[-2]]][:2]
print("New df\n", df)
print("Null Values\n", pd.isnull(df))
print("Total Null Values\n", pd.isnull(df).sum())
print("Not Null Values\n", df.notnull())
print("Last Column Doubled\n", 2 * df[df.columns[-1]])
print("Last Column plus NaN\n", df[df.columns[-1]] + np.nan)
print("Zero filled\n", df.fillna(0))

###########################
print("Date range", pd.date_range('1/1/1900', periods=42, freq='D'))

import sys
try:
    print("Date range", pd.date_range('1/1/1677', periods=4, freq='D'))
except:
    etype, value, _ = sys.exc_info()
    print("Error encountered", etype, value)

offset = pd.DateOffset(seconds=2**33 / 10**9)
mid = pd.to_datetime('1/1/1970')
print("Start valid range", mid - offset)
print("End valid range", mid + offset)

print("Width format", pd.to_datetime(['19021112', '19031230'],
                                     format='%Y%m%d'))
print("Illegal date",
      pd.to_datetime(['1902-11-12', 'not a date'], errors='coerce'))

##########数据透视表
seed(42)
N = 7
df = pd.DataFrame({
    'Weather': ['cold', 'hot', 'cold', 'hot', 'cold', 'hot', 'cold'],
    'Food':
    ['soup', 'soup', 'icecream', 'chocolate', 'icecream', 'icecream', 'soup'],
    'Price':
    10 * rand(N),
    'Number':
    randint(1, 9)
})
print("DataFrame\n", df)
print(pd.pivot_table(df, columns=['Food'], aggfunc=np.sum))
