import warnings # unavailable tickers raise warnings that we'll ignore
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
#import json

ticker = pd.read_csv('Quiz_Ticker_universe.csv', delimiter=',', skiprows=0)
start = dt.datetime(2015, 1, 1)
end = dt.datetime(2016, 5, 10)

dtemp = data.DataReader(ticker['Ticker'], 'yahoo', start, end) #When ticker is not available, warnings will be raised,
                                                         #and NAN filled up in, but the process continues. 
f = dtemp['Close'].dropna(axis=1, how='any')#drop columns that have NAN
print(f)

lnf = np.log(f)
logreturn = lnf - lnf.shift(1)
print(logreturn)

def f_rank(iterable, start=1): # Fractional ranking
    last, fifo = None, []
    for n, item in enumerate(iterable, start):
        if item != last:
            if fifo:
                mean = sum(f[0] for f in fifo) / len(fifo)
                while fifo:
                    yield mean, fifo.pop(0)[1]
        last = item
        fifo.append((n, item))
    if fifo:
        mean = sum(f[0] for f in fifo) / len(fifo)
        while fifo:
            yield mean, fifo.pop(0)[1]
            
# get the rank variable matrix fr
fr = f
for i in range(len(f.columns)):
    tempDF = pd.DataFrame(np.ones(len(f.index)), index = f.iloc[:,i].sort_values().index, columns = ['temp'])
    temp = []
    for rank, price in f_rank(f.iloc[:,i].sort_values()):
        #print('  %3g, %r' % (rank, price))          
        temp.append(rank)
    tempDF['temp'] = temp
    tempDF.sort_index(inplace=True)
    fr.iloc[:,i] = tempDF['temp']
print(fr)

# calculate the pearson's r of the rank variable matrix
nn = len(f.columns)
corr = pd.DataFrame(np.ones((nn,nn)), index = f.columns, columns = f.columns)
for i in range(nn):
    for j in range(nn):
        corr.iloc[i,j] = np.corrcoef(fr.iloc[:,i], fr.iloc[:,j])[0,1]   
print(corr)
corr.to_csv('corr.csv')

dist = 1 - abs(corr)
#distance metric verification
for i in range(nn):
    if dist.iloc[i,i]!=0:
        print(str(i)+","+str(i)+" is not equal to 0")
for i in range(nn):
    for j in range(nn):
        if dist.iloc[i,j] != dist.iloc[j,i]:
            print(str(i)+","+str(j)+" is not equal to "+str(j)+","+str(i))
for i in range(nn):
    for j in range(nn):
        if i != j:
            if dist.iloc[i,j] <= 0:
                print(str(i)+","+str(j)+" <= 0")

NumberOfBreaks=0
for i in range(nn):                     
    for j in range(nn):
        for k in range(nn):
            if dist.iloc[i,j] > dist.iloc[i,k] + dist.iloc[k,j]:#the triangle inequality              
                NumberOfBreaks+=1  
                #print(str(i)+","+str(j)+","+str(k)+" breaks the triangle inequality")
print(NumberOfBreaks)
dist.to_csv('dist.csv')
#print(dist)

#transform dist into a upper triangular matrix to be used in the MST algorithm below
for i in range(nn):
    for j in range(nn):
        if dist.iloc[i,j] != 0:
            dist.iloc[j,i] = 0
        else:
            pass

X = csr_matrix(dist)#compressed sparse row format
Tcsr = minimum_spanning_tree(X)#Kruskal's algorithm
MST = Tcsr.toarray()
MSTdf = pd.DataFrame(MST, index = f.columns, columns = f.columns)
print(MSTdf)
MSTdf.to_json('MST.json', orient='index')
