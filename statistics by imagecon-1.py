import re
value='aaEDA+9  1156Zxdaq  #@#zxfsdfgh'
print(re.sub('[+]', '-', value))
#[^A-Za-z] - only capital and small letter from a to z
#arn -remove arn and give everything
#^arn -only match data
#[a-r] - not a to r and everything
#[0123]-remove 0123
#[0-9]-remove 0 to 9
#[0-5][0-9]- remove 00 to 59 and give everything
#[a-zA-Z]-remove a to z and A to Z
#[+]-remove only +

name='python'
name.endswith('n')
print(name)

a='some'
a=a.replace('some','sam')
print(a)

name.isalpha()
name.isalnum()

#statistics
import numpy as np
z=[10,6,12,3,4,5,2]
z.sort(reverse=False)
np.mean(z)
np.median(z)

from scipy import stats
z=[10,20,20,30,12,3,5,2,6,3,20]
a=stats.mode(z)
print(a)

#measure of spread-iqr
#standard deviation-how standard we deviated everything
#2,4,6,8,10 std is 2
#varience avg squarded deviation about the mean
#if the data points are to far from the mean then there is higher deviation within the data

from scipy import stats
q0=np.quantile(z,0)
q1=np.quantile(z,0.25)
q2=np.quantile(z,0.50)
q3=np.quantile(z,0.75)
q4=np.quantile(z,1)
print(q0,q1,q2,q3,q4)
iqr=q3-q1
print(iqr)
#doubt if i use here quartile the np.auartile will be used

import pandas as pd
data=pd.read_csv('C:/Users\hp\Downloads\mba.csv')
data.head()
data['gmat'].mean()
data.mean()
data['gmat'].median()
data['gmat'].std()
data['gmat'].var()
range_of_data=max(data['gmat'])-min(data['gmat'])
print(range_of_data)
data.quantile(0.5)

#probability
#in probability there is many distribution of data
#normal distribution or gaussion distribution or bell curve
#binommial distribution
#poission distribution

#normal distribution for continuous data
#pdf (dense of data)
from scipy.stats import norm
import seaborn as sns
import numpy as np
data=np.arange(1,100,1)
data.mean()
data.std()
pdf=norm.pdf(data,loc=50.0,scale=28.57)
sns.lineplot(data,pdf,color='green')

#normal cumulative(sorted of data) density distribution
cdf=norm.cdf(data,loc=50.0,scale=28.57)
sns.lineplot(data,cdf,color='black',alpha=0.9)

#example- what is the probability that how many student
#get above the 84 marks
cdfn=norm(loc=90,scale=15).cdf(84)
print(cdfn)#lower tail less than 84
prob=1-cdfn# above 84 upper tail
print(prob)

#probability to get the marks between 60 to 80
val_60=norm(loc=60,scale=15).cdf(60)
val_80=norm(loc=60,scale=15).cdf(80)
prob=val_80-val_60
print(prob)

#random normal distribution
data_normal=norm.rvs(size=100,loc=50.5,scale=28.87)
ax=sns.distplot(data_normal,bins=10,kde=True,color='skyblue'
                , hist_kws={'linewidth':10, 'alpha':0.5})
ax.set(x_label='Normal distribution', y_label='frequency') 

#binomial distribution- discrete data 
from scipy.stats import binom
import matplotlib.pyplot as plt
import seaborn as sns
pb=binom(n=12,p=1)#doubt why you give here probability value???
x=np.arange(1,13)
pmf=pb.pmf(x)
plt.vlines(x,0,pmf,color='k',linestyles='-', lw=5)
plt.xlabel('intervals')
plt.ylabel('probability')
plt.show()
#not show any output

#binomial cumulative density function
cdfb=binom(n=12,p=0.25).cdf(8)#prob less than 8
prob=1-cdfb
print(prob)#above 8

#4 to 8
import numpy as np
cdfl=binom(n=12,p=0.25).cdf(4)
cdfu=binom(n=12,p=0.25).cdf(8)
prob=cdfu-cdfl
print(prob)

#random binomial distribution
data_binom=binom.rvs(n=12,p=0.25,size=10)
ax=sns.distplot(data_binom,bins=5,kde=True,color='skyblue',
                hist_kws={'linewidth':15,'alpha':1})
ax.set(xlabel='binomial distribution',ylabel='frequency')
#what i understand with this figure doubt???

#poisson distribution 
#time and frequency of an event is fix
from scipy.stats import poisson
x=np.arange(1,11)
pmf=poisson.pmf(x,4)
plt.vlines(x,0,pmf,color='k',linestyles='solid',lw=6)
plt.ylabel('probability')
plt.xlabel('intervals')

#poission cumulative distribution
#if there are 12 cars crossing a bridge per minutes on 
#avg find the probability of having 17 or more cars 
#crossing the bridge in particular minute.
cdfp=poisson.cdf(17,mu=12)#less than 17 cars in a minute
#why we are not use pdf here in this
prob=1-cdfp# uppar 17 cars in a minute
print(prob)
 #find probability of 17 to 20 cars in minute
 cdfpl=poisson.cdf(17,mu=12)
 cdfpu=poisson.cdf(20,mu=12)
 prob=cdfpu-cdfpl
 print(prob)

#random poisson distribution
data_poisson=poisson.rvs(mu=12,size=10)
ax=sns.distplot(data_poisson,bins=15,kde=True,color='skyblue',
                hist_kws={'linewidth':15,'alpha':1})

ax.set(xlabel='poisson distribution',ylabel='frequency')
#pls explain graph









